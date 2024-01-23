# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 20:47:29 2023

@author: Aneeta
"""

import speech_recognition as sr

def speech_to_text():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Use the default microphone as the source for audio
    with sr.Microphone() as source:
        print("Listening... Say something!")
        audio = recognizer.listen(source)

    try:
        # Use the Google Web Speech API to recognize the audio
        text = recognizer.recognize_google(audio)
        print("You said:", text)
    except sr.UnknownValueError:
        print("Could not understand audio. Please try again.")
    except sr.RequestError as e:
        print("Error making the request. Check your internet connection. {0}".format(e))

if __name__ == "__main__":
    speech_to_text()