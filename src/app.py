import eel
import speech_recognition as sr
import threading
import pyttsx3
import pyautogui
import time
from gesture_control import GestureController

gesture_running = False
gesture_controller = None
engine = pyttsx3.init()

# ========================= SPEECH =========================
def speak(text):
    """Speak text using pyttsx3."""
    print(f"Proton: {text}")
    engine.say(text)
    engine.runAndWait()


# ========================= GESTURE CONTROL =========================
def start_gesture_navigation():
    """Starts gesture control in a background thread."""
    global gesture_running, gesture_controller
    if gesture_running:
        msg = "Gesture navigation is already running."
        eel.addAppMsg(msg)
        speak(msg)
        return

    gesture_running = True
    msg = "Starting gesture navigation."
    eel.addAppMsg(msg)
    speak(msg)

    def run_gesture():
        global gesture_running, gesture_controller
        try:
            gesture_controller = GestureController()
            gesture_controller.start()
        except Exception as e:
            error_msg = f"Gesture navigation stopped due to error: {e}"
            eel.addAppMsg(error_msg)
            speak("An error occurred while running gesture navigation.")
        finally:
            gesture_running = False
            gesture_controller = None
            eel.addAppMsg("Gesture navigation stopped.")
            speak("Gesture navigation stopped.")

    threading.Thread(target=run_gesture, daemon=True).start()


def stop_gesture_navigation():
    """Stops gesture navigation."""
    global gesture_running, gesture_controller
    if not gesture_running:
        msg = "Gesture navigation is not running."
        eel.addAppMsg(msg)
        speak(msg)
        return

    if gesture_controller:
        eel.addAppMsg("Stopping gesture navigation...")
        speak("Stopping gesture navigation.")
        gesture_controller.stop()

    gesture_running = False
    gesture_controller = None


# ========================= FEATURES =========================
def take_screenshot():
    """Takes a screenshot and saves it with a timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    pyautogui.screenshot(filename)
    eel.addAppMsg(f"📸 Screenshot saved as {filename}")
    speak("Screenshot captured and saved.")


# ========================= COMMAND HANDLER =========================
def process_command(command):
    """Process user’s voice commands."""
    command = command.lower().strip()
    print(f"Processing Command: {command}")

    if "hello proton" in command:
        response = "Hello there! Nice to see you."
        eel.addAppMsg(response)
        speak(response)

    elif "start gesture navigation" in command:
        start_gesture_navigation()

    elif "stop gesture navigation" in command:
        stop_gesture_navigation()

    elif "zoom in" in command:
        response = "You can zoom in by bringing your thumb and index finger closer together."
        eel.addAppMsg(response)
        speak(response)

    elif "zoom out" in command:
        response = "You can zoom out by moving your thumb and index finger apart."
        eel.addAppMsg(response)
        speak(response)

    elif "take screenshot" in command or "capture screen" in command:
        take_screenshot()

    else:
        response = "I'm not programmed to do that yet."
        eel.addAppMsg(response)
        speak(response)


# ========================= VOICE INPUT =========================
@eel.expose
def mic_triggered():
    """Listens for voice input and processes it."""
    print("🎤 Mic Triggered...")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        eel.addAppMsg("Listening...")
        speak("Listening")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print(f"User said: {text}")
        eel.addUserMsg(text)
        process_command(text)
    except sr.UnknownValueError:
        eel.addAppMsg("Sorry, I didn't catch that.")
        speak("Sorry, I didn't catch that.")
    except sr.RequestError:
        eel.addAppMsg("Speech service error.")
        speak("Speech service error.")


# ========================= INITIAL GREETING =========================
@eel.expose
def greet_user():
    greeting = "Hello! I am Proton. How can I help you?"
    eel.addAppMsg(greeting)
    speak(greeting)


# ========================= APP ENTRY =========================
class ChatBot:
    @staticmethod
    def start():
        eel.init("../web")
        eel.start("index.html", size=(1000, 600))


# ========================= MAIN =========================
if __name__ == "__main__":
    ChatBot.start()
