# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 20:49:24 2023

@author: Aneeta
"""

import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# Capture audio from the microphone
with sr.Microphone() as source:
    print("Say something in Kannada:")
    audio = recognizer.listen(source)

try:
    # Recognize the speech
    recognized_text = recognizer.recognize_google(audio, language="kn-IN")
    print("You said:", recognized_text)
except sr.UnknownValueError:
    print("Sorry, I could not understand what you said.")
except sr.RequestError as e:
    print("Error connecting to the Google API; {0}".format(e))
