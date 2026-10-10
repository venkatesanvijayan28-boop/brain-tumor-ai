import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()


def speak_result(text):
    """
    Speak the prediction result using text-to-speech.

    Args:
        text (str): The text to speak aloud.
    """
    try:
        engine.setProperty('rate', 150)    # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

        # Try to use a female voice if available
        voices = engine.getProperty('voices')
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id)

        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Voice alert error: {e}")
