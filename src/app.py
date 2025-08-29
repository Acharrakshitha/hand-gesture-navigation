import eel
import speech_recognition as sr
import threading
import pyttsx3
from gesture_control import GestureController  # <-- IMPORT GESTURE CONTROL

gesture_running = False
gesture_controller = None
engine = pyttsx3.init()

# --- Speak Function ---
def speak(text):
    engine.say(text)
    engine.runAndWait()

# --- Start Gesture Navigation ---
def start_gesture_navigation():
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
        gesture_controller = GestureController()
        gesture_controller.start()
        # Reset state when window closes
        gesture_running = False
        gesture_controller = None
        eel.addAppMsg("Gesture navigation stopped.")   # <-- only final confirmation
        speak("Gesture navigation stopped.")

    threading.Thread(target=run_gesture, daemon=True).start()

# --- Stop Gesture Navigation ---
def stop_gesture_navigation():
    global gesture_running, gesture_controller
    if not gesture_running:
        msg = "Gesture navigation is not running."
        eel.addAppMsg(msg)
        speak(msg)
        return

    if gesture_controller:
        eel.addAppMsg("Stopping gesture navigation...")   # <-- shown first
        speak("Stopping gesture navigation")
        gesture_controller.stop()  # This will eventually trigger "stopped" in run_gesture()

    gesture_running = False
    gesture_controller = None

# --- Handle Commands ---
def process_command(command):
    command = command.lower()
    if "hello proton" in command:
        response = "Hello there! Nice to see you."
        eel.addAppMsg(response)
        speak(response)
    elif "start gesture navigation" in command:
        start_gesture_navigation()
    elif "stop gesture navigation" in command:
        stop_gesture_navigation()
    else:
        response = "I am not functioned to do this."
        eel.addAppMsg(response)
        speak(response)

# --- Microphone Listener ---
@eel.expose
def mic_triggered():
    print("Mic Triggered...")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        eel.addAppMsg("Listening... 🎤")
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

# --- Greet User After Page Load ---
@eel.expose
def greet_user():
    greeting = "Hello! I am Proton. How may I help you?"
    eel.addAppMsg(greeting)
    speak(greeting)

# --- Start App ---
class ChatBot:
    @staticmethod
    def start():
        eel.init("../web")
        eel.start("index.html", size=(1000, 600))
