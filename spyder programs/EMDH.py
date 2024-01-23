import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from hurst import compute_Hc, random_walk
import speech_recognition as sr
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the speech signal
audio_path = input("D:\Aneeta_Phd\Speech Datasets\Enhanced\enhanced_signal.wav")
speech_signal, sr_speech = sf.read(audio_path)

# Perform Empirical Mode Decomposition (EMD) using Hurst-based mode selection (EMDH)
def perform_emd(signal):
    emd_results = []
    for _ in range(num_emd_modes):
        hurst = compute_Hc(signal, kind='random_walk', simplified=True)
        if hurst > threshold:
            emd_results.append(signal)
        signal = signal - emd_results[-1]
    return emd_results

num_emd_modes = 5
threshold = 0.5
emd_results = perform_emd(speech_signal)

# Combine selected EMD modes
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

# Define and train a CNN model for speech recognition
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Assuming you have labeled speech data for training
input_shape = (num_emd_modes, len(combined_emd))
num_classes = 2  # Modify based on your classes
model = build_cnn_model(input_shape, num_classes)

# Train the CNN model using your labeled speech data
# ...

# Visualize the decomposed signals and the combined EMD signal
plt.figure(figsize=(10, 6))
plt.subplot(len(emd_results) + 2, 1, 1)
librosa.display.waveshow(speech_signal, sr=sr_speech)
plt.title("Original Speech Signal")

for idx, emd_signal in enumerate(emd_results):
    plt.subplot(len(emd_results) + 2, 1, idx + 2)
    librosa.display.waveshow(emd_signal, sr=sr_speech)
    plt.title(f"Selected EMD Mode {idx + 1}")

plt.subplot(len(emd_results) + 2, 1, len(emd_results) + 2)
librosa.display.waveshow(combined_emd, sr=sr_speech)
plt.title("Combined EMD Components")

plt.tight_layout()
plt.show()
