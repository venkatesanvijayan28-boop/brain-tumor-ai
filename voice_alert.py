import threading
import pyttsx3

try:
    import pythoncom
except ImportError:
    pythoncom = None


def speak_result(text):
    """
    Speak the prediction result using text-to-speech.

    Args:
        text (str): The text to speak aloud.
    """
    print(f"[Voice Alert] {text}")

    def _speak():
        try:
            if pythoncom:
                pythoncom.CoInitialize()

            engine = pyttsx3.init()
            engine.setProperty('rate', 150)    # Speed of speech
            engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

            # Try to use a female voice if available
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)

            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"Voice alert error: {e}")
        finally:
            if pythoncom:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    # Run in background daemon thread so it doesn't block the web server
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()

