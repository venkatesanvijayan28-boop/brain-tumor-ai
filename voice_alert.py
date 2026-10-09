import sys
import os

def speak_result(text):
    """
    Log voice text. Voice synthesis is performed client-side in browser.
    Prevents server hanging or blocking on headless cloud environments (Render).

    Args:
        text (str): The text to be spoken.
    """
    print(f"[Voice Alert] {text}")
    # Only attempt pyttsx3 if running on a local desktop with display/audio attached
    if sys.platform == 'win32' and os.environ.get('ENABLE_SERVER_VOICE', 'false').lower() == 'true':
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[Voice Alert] Local TTS skipped: {e}")


