# Imports
import cv2
import mediapipe as mp
import pyautogui
import math
import time
import os
from datetime import datetime
from enum import IntEnum
from google.protobuf.json_format import MessageToDict
import screen_brightness_control as sbcontrol

pyautogui.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# ----------------------------- ENUMERATIONS -----------------------------
class Gest(IntEnum):
    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16
    PALM = 31
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36
    ZOOM = 37
    SWIPE = 38
    FIVE = 39  # ✋ Added for screenshot gesture


class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1


# ----------------------------- HAND RECOGNITION -----------------------------
class HandRecog:
    def __init__(self, hand_label):
        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label
        self.prev_x = None
        self.prev_time = time.time()

    def update_hand_result(self, hand_result):
        self.hand_result = hand_result

    def get_signed_dist(self, point):
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x) ** 2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y) ** 2
        return math.sqrt(dist) * sign

    def get_dist(self, point):
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x) ** 2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y) ** 2
        return math.sqrt(dist)

    def get_dz(self, point):
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)

    def set_finger_state(self):
        if self.hand_result is None:
            return
        points = [[8, 5, 0], [12, 9, 0], [16, 13, 0], [20, 17, 0]]
        self.finger = 0
        for point in points:
            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])
            ratio = round(dist / dist2 if dist2 != 0 else dist / 0.01, 1)
            self.finger = self.finger << 1
            if ratio > 0.5:
                self.finger = self.finger | 1

    def detect_swipe(self):
        """Detects left/right swipe by measuring x-velocity of hand center."""
        if self.hand_result is None:
            return None

        cx = self.hand_result.landmark[9].x
        cy = self.hand_result.landmark[9].y
        cur_time = time.time()

        if self.prev_x is None:
            self.prev_x = cx
            self.prev_time = cur_time
            return None

        dt = cur_time - self.prev_time
        dx = cx - self.prev_x
        dy = cy - self.hand_result.landmark[9].y

        self.prev_x = cx
        self.prev_time = cur_time

        if abs(dx) < 0.08 or abs(dy) > 0.05 or dt == 0:
            return None

        velocity = dx / dt
        if abs(velocity) > 1.2:
            return "RIGHT" if velocity > 0 else "LEFT"
        return None

    def get_gesture(self):
        if self.hand_result is None:
            return Gest.PALM

        current_gesture = Gest.PALM
        thumb_tip = self.hand_result.landmark[4]
        index_tip = self.hand_result.landmark[8]
        pinch_distance = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)

        if pinch_distance < 0.12:
            current_gesture = Gest.ZOOM
        elif self.finger == 15:  # All five fingers open (binary 1111)
            current_gesture = Gest.FIVE
        elif self.finger in [Gest.LAST3, Gest.LAST4] and self.get_dist([8, 4]) < 0.05:
            current_gesture = Gest.PINCH_MINOR if self.hand_label == HLabel.MINOR else Gest.PINCH_MAJOR
        elif self.finger == Gest.FIRST2:
            dist1 = self.get_dist([8, 12])
            dist2 = self.get_dist([5, 9])
            ratio = dist1 / dist2
            if ratio > 1.7:
                current_gesture = Gest.V_GEST
            elif self.get_dz([8, 12]) < 0.1:
                current_gesture = Gest.TWO_FINGER_CLOSED
            else:
                current_gesture = Gest.MID
        else:
            current_gesture = self.finger

        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0
        self.prev_gesture = current_gesture

        if self.frame_count > 4:
            self.ori_gesture = current_gesture
        return self.ori_gesture


# ----------------------------- CONTROLLER -----------------------------
class Controller:
    tx_old = 0
    ty_old = 0
    flag = False
    grabflag = False
    pinchmajorflag = False
    pinchminorflag = False
    prev_hand = None
    pinch_threshold = 0.3

    prev_distance = None
    zoom_smooth = 0.0
    zoom_active = False
    ZOOM_ACTIVATE_DIST = 0.12
    ZOOM_RELEASE_DIST = 0.17
    ZOOM_IN_THRESHOLD = 0.02
    ZOOM_OUT_THRESHOLD = -0.02

    last_swipe_time = 0
    SWIPE_COOLDOWN = 1.5

    last_play_pause_time = 0
    PLAY_PAUSE_COOLDOWN = 1.5

    last_screenshot_time = 0
    SCREENSHOT_COOLDOWN = 2.0  # seconds between screenshots

    @staticmethod
    def handle_zoom(hand_result):
        thumb_tip = hand_result.landmark[4]
        index_tip = hand_result.landmark[8]
        distance = math.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 +
            (thumb_tip.y - index_tip.y) ** 2 +
            (thumb_tip.z - index_tip.z) ** 2
        )

        if not Controller.zoom_active:
            if distance < Controller.ZOOM_ACTIVATE_DIST:
                Controller.zoom_active = True
                Controller.prev_distance = distance
            else:
                return
        else:
            if distance > Controller.ZOOM_RELEASE_DIST:
                Controller.zoom_active = False
                Controller.prev_distance = None
                return

        if Controller.prev_distance is None:
            Controller.prev_distance = distance
            return

        diff = distance - Controller.prev_distance
        Controller.zoom_smooth = 0.7 * Controller.zoom_smooth + 0.3 * diff

        if Controller.zoom_smooth > Controller.ZOOM_IN_THRESHOLD:
            pyautogui.hotkey('ctrl', '+')
            print("🔍 Zoom In")
            Controller.prev_distance = distance
        elif Controller.zoom_smooth < Controller.ZOOM_OUT_THRESHOLD:
            pyautogui.hotkey('ctrl', '-')
            print("🔎 Zoom Out")
            Controller.prev_distance = distance

    @staticmethod
    def handle_screenshot():
        """Capture a screenshot and save to 'screenshots' folder."""
        current_time = time.time()
        if current_time - Controller.last_screenshot_time < Controller.SCREENSHOT_COOLDOWN:
            return

        Controller.last_screenshot_time = current_time
        os.makedirs("screenshots", exist_ok=True)
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.png")
        filepath = os.path.join("screenshots", filename)
        pyautogui.screenshot(filepath)
        print(f"📸 Screenshot saved: {filepath}")

    @staticmethod
    def handle_swipe(direction):
        current_time = time.time()
        if current_time - Controller.last_swipe_time < Controller.SWIPE_COOLDOWN:
            return
        Controller.last_swipe_time = current_time

        if direction == "RIGHT":
            print("👉 Swipe Right → Next Window")
            pyautogui.keyDown('alt')
            pyautogui.press('tab')
            pyautogui.keyUp('alt')
        elif direction == "LEFT":
            print("👈 Swipe Left → Previous Window")
            pyautogui.keyDown('shift')
            pyautogui.keyDown('alt')
            pyautogui.press('tab')
            pyautogui.keyUp('alt')
            pyautogui.keyUp('shift')

    @staticmethod
    def handle_play_pause():
        current_time = time.time()
        if current_time - Controller.last_play_pause_time > Controller.PLAY_PAUSE_COOLDOWN:
            pyautogui.press('space')
            Controller.last_play_pause_time = current_time
            print("🎵 Play/Pause toggled")

    @staticmethod
    def get_position(hand_result):
        point = 9
        position = [hand_result.landmark[point].x, hand_result.landmark[point].y]
        sx, sy = pyautogui.size()
        x_old, y_old = pyautogui.position()
        x = int(position[0] * sx)
        y = int(position[1] * sy)
        if Controller.prev_hand is None:
            Controller.prev_hand = [x, y]
        delta_x = x - Controller.prev_hand[0]
        delta_y = y - Controller.prev_hand[1]
        distsq = delta_x ** 2 + delta_y ** 2
        ratio = 1
        Controller.prev_hand = [x, y]
        if distsq <= 25:
            ratio = 0
        elif distsq <= 900:
            ratio = 0.07 * (distsq ** 0.5)
        else:
            ratio = 2.1
        x, y = x_old + delta_x * ratio, y_old + delta_y * ratio
        return (x, y)

    @staticmethod
    def handle_controls(gesture, hand_result, swipe_dir=None, image=None):
        x, y = None, None
        if gesture != Gest.PALM:
            x, y = Controller.get_position(hand_result)

        if gesture == Gest.ZOOM:
            Controller.handle_zoom(hand_result)
            if image is not None:
                cv2.putText(image, "Zoom Mode Active", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            return

        if swipe_dir:
            Controller.handle_swipe(swipe_dir)
            if image is not None:
                cv2.putText(image, f"Swipe {swipe_dir}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            return

        if gesture == Gest.FIST:
            Controller.handle_play_pause()
            if image is not None:
                cv2.putText(image, "Play/Pause", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            return

        if gesture == Gest.FIVE:
            Controller.handle_screenshot()
            if image is not None:
                cv2.putText(image, "📸 Screenshot Taken", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            return

        if gesture != Gest.FIST and Controller.grabflag:
            Controller.grabflag = False
            pyautogui.mouseUp(button="left")

        if gesture == Gest.V_GEST:
            Controller.flag = True
            pyautogui.moveTo(x, y, duration=0.1)

        elif gesture == Gest.MID and Controller.flag:
            pyautogui.click()
            Controller.flag = False

        elif gesture == Gest.INDEX and Controller.flag:
            pyautogui.click(button="right")
            Controller.flag = False

        elif gesture == Gest.TWO_FINGER_CLOSED and Controller.flag:
            pyautogui.doubleClick()
            Controller.flag = False


# ----------------------------- MAIN GESTURE CONTROLLER -----------------------------
class GestureController:
    gc_mode = 0
    cap = None
    CAM_HEIGHT = None
    CAM_WIDTH = None
    hr_major = None
    hr_minor = None
    dom_hand = True

    def __init__(self):
        GestureController.gc_mode = 1
        GestureController.cap = cv2.VideoCapture(0)
        GestureController.CAM_HEIGHT = GestureController.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        GestureController.CAM_WIDTH = GestureController.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def classify_hands(results):
        left, right = None, None
        try:
            handedness_dict = MessageToDict(results.multi_handedness[0])
            if handedness_dict["classification"][0]["label"] == "Right":
                right = results.multi_hand_landmarks[0]
            else:
                left = results.multi_hand_landmarks[0]
        except:
            pass
        try:
            handedness_dict = MessageToDict(results.multi_handedness[1])
            if handedness_dict["classification"][0]["label"] == "Right":
                right = results.multi_hand_landmarks[1]
            else:
                left = results.multi_hand_landmarks[1]
        except:
            pass
        if GestureController.dom_hand:
            GestureController.hr_major = right
            GestureController.hr_minor = left
        else:
            GestureController.hr_major = left
            GestureController.hr_minor = right

    def start(self):
        handmajor = HandRecog(HLabel.MAJOR)
        handminor = HandRecog(HLabel.MINOR)

        cv2.namedWindow("Gesture Controller", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Gesture Controller", cv2.WND_PROP_TOPMOST, 1)

        with mp_hands.Hands(max_num_hands=2,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as hands:
            while GestureController.cap.isOpened() and GestureController.gc_mode:
                success, image = GestureController.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                swipe_dir = None

                if results.multi_hand_landmarks:
                    GestureController.classify_hands(results)
                    handmajor.update_hand_result(GestureController.hr_major)
                    handminor.update_hand_result(GestureController.hr_minor)
                    handmajor.set_finger_state()
                    handminor.set_finger_state()

                    swipe_dir = handmajor.detect_swipe()

                    gest_minor = handminor.get_gesture()
                    if gest_minor == Gest.PINCH_MINOR:
                        Controller.handle_controls(gest_minor, handminor.hand_result)
                    else:
                        gest_major = handmajor.get_gesture()
                        Controller.handle_controls(gest_major, handmajor.hand_result, swipe_dir, image)

                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                else:
                    Controller.prev_hand = None

                cv2.imshow("Gesture Controller", image)
                cv2.setWindowProperty("Gesture Controller", cv2.WND_PROP_TOPMOST, 1)
                if cv2.waitKey(5) & 0xFF == 13:
                    break

        GestureController.cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def stop():
        GestureController.gc_mode = 0
        if GestureController.cap and GestureController.cap.isOpened():
            GestureController.cap.release()
        cv2.destroyAllWindows()
