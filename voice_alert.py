import sys
import os


def speak_result(text):
    """
    Log voice text. Voice synthesis is performed client-side in the browser
    via Web Speech API to eliminate duplicate overlapping voices and ensure
    a single female voice with functional on/off control.

    Args:
        text (str): The text to be spoken.
    """
    print(f"[Voice Alert] {text}")


