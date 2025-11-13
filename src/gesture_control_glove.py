import numpy as np
import cv2
import cv2.aruco as aruco
import os
import glob
import math
import pyautogui
import time

# ----------------------- Marker Detection -----------------------
class Marker:
    def __init__(self, dict_type = aruco.DICT_4X4_50, thresh_constant = 1):
        self.aruco_dict = aruco.Dictionary_get(dict_type)
        self.parameters = aruco.DetectorParameters_create()
        self.parameters.adaptiveThreshConstant = thresh_constant
        self.corners = None
        self.marker_x2y = 1
        self.mtx, self.dist = Marker.calibrate()
    
    def calibrate():
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        objpoints = []
        imgpoints = []
        path = os.path.dirname(os.path.abspath(__file__))
        p1 = path + r'\calib_images\checkerboard\*.jpg'
        images = glob.glob(p1)
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
                img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
                
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        return mtx, dist
    
    def detect(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_frame, self.aruco_dict, parameters = self.parameters)
        if np.all(ids != None):
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(self.corners, 0.05, self.mtx, self.dist)
        else:
            self.corners = None
    
    def is_detected(self):
        if self.corners:
            return True
        return False
    
    def draw_marker(self, frame):
        aruco.drawDetectedMarkers(frame, self.corners)

# ----------------------- Utility Functions -----------------------
def ecu_dis(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def find_HSV(samples):
    try:
        color = np.uint8([ samples ])
    except:
        color = np.uint8([ [[105,105,50]] ])
    hsv_color = cv2.cvtColor(color,cv2.COLOR_RGB2HSV)
    return hsv_color

def draw_box(frame, points, color=(0,255,127)):
    if points:
        frame = cv2.line(frame, points[0], points[1], color, thickness=2, lineType=8)
        frame = cv2.line(frame, points[1], points[2], color, thickness=2, lineType=8)
        frame = cv2.line(frame, points[2], points[3], color, thickness=2, lineType=8)
        frame = cv2.line(frame, points[3], points[0], color, thickness=2, lineType=8)

def in_cam(val, type_):
    if type_ == 'x':
        if val<0: return 0
        if val>GestureController.cam_width: return GestureController.cam_width
    elif type_ == 'y':
        if val<0: return 0
        if val>GestureController.cam_height: return GestureController.cam_height
    return val

# ----------------------- ROI -----------------------
class ROI:
    def __init__(self, roi_alpha1=1.5, roi_alpha2=1.5, roi_beta=2.5, hsv_alpha = 0.3, hsv_beta = 0.5, hsv_lift_up = 0.3):
        self.roi_alpha1 = roi_alpha1
        self.roi_alpha2 = roi_alpha2
        self.roi_beta = roi_beta
        self.roi_corners = None
        self.hsv_alpha = hsv_alpha
        self.hsv_beta = hsv_beta
        self.hsv_lift_up = hsv_lift_up
        self.hsv_corners = None
        self.marker_top = None
        self.hsv_glove = None
        
    def findROI(self, frame, marker):
        rec_coor = marker.corners[0][0]
        c1 = (int(rec_coor[0][0]),int(rec_coor[0][1]))
        c2 = (int(rec_coor[1][0]),int(rec_coor[1][1]))
        c3 = (int(rec_coor[2][0]),int(rec_coor[2][1]))
        c4 = (int(rec_coor[3][0]),int(rec_coor[3][1]))
        
        try:
            marker.marker_x2y = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) / np.sqrt((c3[0]-c2[0])**2 + (c3[1]-c2[1])**2)
        except:
            marker.marker_x2y = 999.0
        
        cx = (c1[0] + c2[0])/2
        cy = (c1[1] + c2[1])/2
        self.marker_top = [cx, cy]
        l = np.absolute(ecu_dis(c1,c4))
        
        try:
            slope_12 = (c1[1]-c2[1])/(c1[0]-c2[0])
        except:
            slope_12 = (c1[1]-c2[1])*999.0 + 0.1
        
        try:
            slope_14 = -1 / slope_12
        except:
            slope_14 = -999.0
        
        sign = 1 if slope_14 < 0 else -1
        
        bot_rx = int(cx + self.roi_alpha2 * l * np.sqrt(1/(1+slope_12**2)))
        bot_ry = int(cy + self.roi_alpha2 * slope_12 * l * np.sqrt(1/(1+slope_12**2)))
        
        bot_lx = int(cx - self.roi_alpha1 * l * np.sqrt(1/(1+slope_12**2)))
        bot_ly = int(cy - self.roi_alpha1 * slope_12 * l * np.sqrt(1/(1+slope_12**2)))
        
        top_lx = int(bot_lx + sign * self.roi_beta * l * np.sqrt(1/(1+slope_14**2)))
        top_ly = int(bot_ly + sign * self.roi_beta * slope_14 * l * np.sqrt(1/(1+slope_14**2)))
        
        top_rx = int(bot_rx + sign * self.roi_beta * l * np.sqrt(1/(1+slope_14**2)))
        top_ry = int(bot_ry + sign * self.roi_beta * slope_14 * l * np.sqrt(1/(1+slope_14**2)))
        
        bot_lx, bot_ly = in_cam(bot_lx, 'x'), in_cam(bot_ly, 'y')
        bot_rx, bot_ry = in_cam(bot_rx, 'x'), in_cam(bot_ry, 'y')
        top_lx, top_ly = in_cam(top_lx, 'x'), in_cam(top_ly, 'y')
        top_rx, top_ry = in_cam(top_rx, 'x'), in_cam(top_ry, 'y')
        
        self.roi_corners = [(bot_lx,bot_ly), (bot_rx,bot_ry), (top_rx,top_ry), (top_lx,top_ly)]
        
    def find_glove_hsv(self, frame, marker):
        rec_coor = marker.corners[0][0]
        c1 = (int(rec_coor[0][0]),int(rec_coor[0][1]))
        c2 = (int(rec_coor[1][0]),int(rec_coor[1][1]))
        c3 = (int(rec_coor[2][0]),int(rec_coor[2][1]))
        c4 = (int(rec_coor[3][0]),int(rec_coor[3][1]))
        
        l = np.absolute(ecu_dis(c1,c4))
        
        try:
            slope_12 = (c1[1]-c2[1])/(c1[0]-c2[0])
        except:
            slope_12 = (c1[1]-c2[1])*999.0 + 0.1
        try:
            slope_14 = -1 / slope_12
        except:
            slope_14 = -999.0
        
        sign = 1 if slope_14 < 0 else -1
               
        bot_rx = int(self.marker_top[0] + self.hsv_alpha * l * np.sqrt(1/(1+slope_12**2)))
        bot_ry = int(self.marker_top[1] - self.hsv_lift_up*l + self.hsv_alpha * slope_12 * l * np.sqrt(1/(1+slope_12**2)))
        
        bot_lx = int(self.marker_top[0] - self.hsv_alpha * l * np.sqrt(1/(1+slope_12**2)))
        bot_ly = int(self.marker_top[1] - self.hsv_lift_up*l - self.hsv_alpha * slope_12 * l * np.sqrt(1/(1+slope_12**2)))
        
        top_lx = int(bot_lx + sign * self.hsv_beta * l * np.sqrt(1/(1+slope_14**2)))
        top_ly = int(bot_ly + sign * self.hsv_beta * slope_14 * l * np.sqrt(1/(1+slope_14**2)))
        
        top_rx = int(bot_rx + sign * self.hsv_beta * l * np.sqrt(1/(1+slope_14**2)))
        top_ry = int(bot_ry + sign * self.hsv_beta * slope_14 * l * np.sqrt(1/(1+slope_14**2)))
        
        region = frame[top_ry:bot_ry , top_lx:bot_rx]
        b, g, r = np.mean(region, axis=(0, 1))
        
        self.hsv_glove = find_HSV([[r,g,b]])
        self.hsv_corners =  [(bot_lx,bot_ly), (bot_rx,bot_ry), (top_rx,top_ry), (top_lx,top_ly)]
        
    def cropROI(self, frame):
        pts = np.array(self.roi_corners)
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        croped = frame[y:y+h, x:x+w].copy()
        pts = pts - pts.min(axis=0)
        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        bg = np.ones_like(croped, np.uint8)*255
        cv2.bitwise_not(bg,bg, mask=mask)
        
        kernelOpen = np.ones((3,3),np.uint8)
        kernelClose = np.ones((5,5),np.uint8)
        
        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        lower_range = np.array([self.hsv_glove[0][0][0]//1-5,50,50])
        upper_range = np.array([self.hsv_glove[0][0][0]//1+5,255,255])
        mask = cv2.inRange(hsv, lower_range, upper_range)
        Opening =cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
        Closing =cv2.morphologyEx(Opening,cv2.MORPH_CLOSE,kernelClose)
        FinalMask = Closing
        return FinalMask

# ----------------------- Glove -----------------------
class Glove:
    def __init__(self):
        self.fingers = 0
        self.arearatio = 0
        self.gesture = 0
        self.pinch_distance = 0
        self.prev_pinch_distance = None
    
    def find_fingers(self, FinalMask):
        conts,h=cv2.findContours(FinalMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        hull = [cv2.convexHull(c) for c in conts]
        try:
            cnt = max(conts, key = lambda x: cv2.contourArea(x))
            epsilon = 0.0005*cv2.arcLength(cnt,True)
            approx= cv2.approxPolyDP(cnt,epsilon,True)
            hull = cv2.convexHull(cnt)
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)
            self.arearatio=((areahull-areacnt)/areacnt)*100
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)
        except:
            defects = None
        
        l=0
        try:
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                a = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
                b = math.sqrt((far[0]-start[0])**2 + (far[1]-start[1])**2)
                c = math.sqrt((end[0]-far[0])**2 + (end[1]-far[1])**2)
                s_ = (a+b+c)/2
                ar = math.sqrt(s_*(s_-a)*(s_-b)*(s_-c))
                d=(2*ar)/a
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                if angle <= 90 and d>30:
                    l += 1
                cv2.line(FinalMask,start, end, [255,255,255], 2)
            l+=1
        except:
            l=0
        self.fingers = l
    
    def find_gesture(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.gesture = 0
        if self.fingers==1:
            if self.arearatio<15:
                cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                self.gesture = 0
            elif self.arearatio<25:
                cv2.putText(frame,'2 fingers',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                self.gesture = 2
            else:
                cv2.putText(frame,'1 finger',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                self.gesture = 1
        elif self.fingers==2:
            cv2.putText(frame,'2 fingers',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            self.gesture = 2
        elif self.fingers==3:
            cv2.putText(frame,'3 fingers',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            self.gesture = 3
        elif self.fingers==4:
            cv2.putText(frame,'4 fingers',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            self.gesture = 4
        elif self.fingers==5:
            cv2.putText(frame,'5 fingers',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            self.gesture = 5
    
    # ----------------------- Pinch Zoom -----------------------
    def detect_pinch(self, FinalMask, frame):
        conts,_ = cv2.findContours(FinalMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(conts) == 0:
            return
        try:
            cnt = max(conts, key=lambda x: cv2.contourArea(x))
            hull = cv2.convexHull(cnt, returnPoints=True)
            
            hull_indices = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull_indices)
            if defects is None:
                return
            
            fingertip_points = []
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                fingertip_points.append(start)
                fingertip_points.append(end)
            
            fingertip_points = sorted(fingertip_points, key=lambda x: x[1])
            if len(fingertip_points) >= 2:
                thumb = fingertip_points[0]
                index = fingertip_points[1]
                self.pinch_distance = ecu_dis(thumb, index)
                
                cv2.circle(frame, thumb, 5, (0,255,0), -1)
                cv2.circle(frame, index, 5, (0,255,0), -1)
                cv2.line(frame, thumb, index, (0,255,0), 2)
        except:
            pass
    
    def apply_zoom(self):
        if self.prev_pinch_distance is None:
            self.prev_pinch_distance = self.pinch_distance
            return
        
        delta = self.pinch_distance - self.prev_pinch_distance
        if abs(delta) > 5:
            if delta > 0:
                pyautogui.hotkey('ctrl', '+')  # Zoom in
            else:
                pyautogui.hotkey('ctrl', '-')  # Zoom out
        self.prev_pinch_distance = self.pinch_distance

# ----------------------- Mouse Controller -----------------------
class MouseController:
    def move_mouse(self, frame, marker_top, gesture):
        try:
            x = int(marker_top[0]*1.5)
            y = int(marker_top[1]*1.5)
            pyautogui.moveTo(x, y)
        except:
            pass

# ----------------------- Gesture Controller -----------------------
class GestureController:
    cam_width = 640
    cam_height = 480
    aru_marker = Marker()
    hand_roi = ROI()
    glove = Glove()
    mouse = MouseController()
    
    @staticmethod
    def start():
        cap = cv2.VideoCapture(0)
        cap.set(3, GestureController.cam_width)
        cap.set(4, GestureController.cam_height)
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            
            GestureController.aru_marker.detect(frame)
            if GestureController.aru_marker.is_detected():
                GestureController.hand_roi.findROI(frame, GestureController.aru_marker)
                GestureController.hand_roi.find_glove_hsv(frame, GestureController.aru_marker)
                FinalMask = GestureController.hand_roi.cropROI(frame)
                
                GestureController.glove.find_fingers(FinalMask)
                GestureController.glove.find_gesture(frame)
                
                # ---- Pinch-to-Zoom ----
                GestureController.glove.detect_pinch(FinalMask, frame)
                GestureController.glove.apply_zoom()
                
                GestureController.mouse.move_mouse(frame, GestureController.hand_roi.marker_top, GestureController.glove.gesture)
            
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# ----------------------- Main -----------------------
if __name__ == "__main__":
    GestureController.start()
