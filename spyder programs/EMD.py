import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import speech_recognition as sr
import os

# Load the speech signal
speech_signal, sr_speech = sf.read("D:\Aneeta_Phd\Speech Datasets\Enhanced\enhanced_signal.wav")

# Perform Empirical Mode Decomposition (EMD) using Librosa
def perform_emd(signal):
    emd_result = librosa.effects.hpss(signal)  # Using Harmonic-Percussive Source Separation as an example
    return emd_result

emd_results = perform_emd(speech_signal)

# Combine EMD components
combined_emd = np.sum(emd_results, axis=0)

# Initialize the recognizer
recognizer = sr.Recognizer()

# Convert the combined EMD signal to an audio file
combined_emd_filename = "combined_emd.wav"
sf.write(combined_emd_filename, combined_emd, sr_speech)

# Recognize speech from the combined EMD signal
with sr.AudioFile(combined_emd_filename) as source:
    audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        print(f"Recognized Speech from Combined EMD: {text}")
    except sr.UnknownValueError:
        print("Could not understand speech from combined EMD")
    except sr.RequestError as e:
        print(f"Recognition request failed for combined EMD: {e}")

# Remove the temporary combined EMD file
os.remove(combined_emd_filename)

# Visualize the decomposed signals
plt.figure(figsize=(10, 6))
plt.subplot(len(emd_results) + 2, 1, 1)
librosa.display.waveshow(speech_signal, sr=sr_speech)
plt.title("Original Speech Signal")

for idx, emd_signal in enumerate(emd_results):
    plt.subplot(len(emd_results) + 2, 1, idx + 2)
    librosa.display.waveshow(emd_signal, sr=sr_speech)
    plt.title(f"EMD Component {idx + 1}")

plt.subplot(len(emd_results) + 2, 1, len(emd_results) + 2)
librosa.display.waveshow(combined_emd, sr=sr_speech)
plt.title("Combined EMD Components")

plt.tight_layout()
plt.show()
