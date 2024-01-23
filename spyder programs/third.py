# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:13:57 2023

@author: Aneeta
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def detect_silence(signal, sr, threshold=0.01, min_silence_duration=0.1):
    """
    Detects silent segments in an audio signal.

    Parameters:
        signal (numpy.ndarray): The input audio signal.
        sr (int): The sampling rate of the audio signal.
        threshold (float): The amplitude threshold below which a segment is considered silent.
        min_silence_duration (float): The minimum duration (in seconds) of a silent segment.

    Returns:
        numpy.ndarray: An array of indices corresponding to the start and end of silent segments.
    """
    split_points = np.where(np.abs(signal) > threshold)[0]
    silent_segments = np.split(split_points, np.where(np.diff(split_points) > 1.0)[0] + 1)
    silent_segments = [seg for seg in silent_segments if len(seg) > min_silence_duration * sr]
    return np.array([(seg[0], seg[-1]) for seg in silent_segments])


def spectral_subtraction(signal, noise, alpha=1):
    """
    Perform spectral subtraction to enhance a noisy speech signal.

    Parameters:
        signal (numpy.ndarray): The input noisy speech signal.
        noise (numpy.ndarray): The noise signal to be subtracted from the input signal.
        alpha (float): A scaling factor for the amount of noise reduction.

    Returns:
        numpy.ndarray: The enhanced speech signal.
    """
    # Ensure both signal and noise have the same length
    max_length = max(len(signal), len(noise))
    signal = np.pad(signal, (0, max_length - len(signal)))
    noise = np.pad(noise, (0, max_length - len(noise)))

    # Calculate the magnitude spectrogram of the signal and noise
    signal_spec = np.abs(librosa.stft(signal))
    noise_spec = np.abs(librosa.stft(noise))

    # Perform Spectral Subtraction
    enhanced_spec = np.maximum(signal_spec - alpha * noise_spec, 0)

    # Inverse STFT to obtain the enhanced signal
    enhanced_signal = librosa.istft(enhanced_spec)

    return enhanced_signal

def create_denoising_autoencoder(input_shape):
    model = keras.Sequential([
        layers.Input(input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(input_shape[-1], activation='sigmoid')  # Use input_shape[-1] for the output dimension
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == "__main__":
    dysarthric_speech, sr_dysarthric = sf.read(r'D:\Aneeta_Phd\Speech Datasets\English\With Dysarthria\M\M01\Session1\wav_arrayMic\0001.wav')
    noise_segments = detect_silence(dysarthric_speech, sr_dysarthric)
    noise = np.concatenate([dysarthric_speech[start:end] for start, end in noise_segments])
    alpha = 2.0
    enhanced_dysarthric = spectral_subtraction(dysarthric_speech, noise, alpha)

    dysarthric_speech = dysarthric_speech / np.max(np.abs(dysarthric_speech))
    enhanced_dysarthric = enhanced_dysarthric / np.max(np.abs(enhanced_dysarthric))

    # The input_shape should be a 2D shape (num_samples, num_features)
    input_shape = (dysarthric_speech.shape[0], 1)  # Reshape to a 2D array with 1 feature
    model = create_denoising_autoencoder(input_shape)

    # Expand dimensions for the denoising autoencoder training
    dysarthric_speech = np.expand_dims(dysarthric_speech, axis=-1)
    enhanced_dysarthric = np.expand_dims(enhanced_dysarthric, axis=-1)

    model.fit(dysarthric_speech, dysarthric_speech, epochs=50, batch_size=32, validation_split=0.2)
    enhanced_dysarthric_deep_learning = model.predict(enhanced_dysarthric)

sf.write('enhanced_dysarthric.wav', enhanced_dysarthric_deep_learning, sr_dysarthric)



    # ... (same visualization code as before)
    # Visualize the original and enhanced dysarthric speech signals (optional)
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.title('Original Dysarthric Speech')
plt.plot(dysarthric_speech)
plt.savefig('dysarthric_plot.png')
plt.subplot(2, 1, 2)
plt.title('Enhanced Dysarthric Speech (Deep Learning)')
plt.plot(enhanced_dysarthric_deep_learning)
plt.tight_layout()
plt.show()
plt.savefig('plot.png')





plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.title('Spectrogram of Original Dysarthric Speech')
plt.specgram(dysarthric_speech, Fs=sr_dysarthric, cmap='viridis')
plt.colorbar(format='%+2.0f dB')

    # Calculate and plot the spectrogram of the enhanced dysarthric speech
plt.subplot(3, 1, 2)
plt.title('Spectrogram of Enhanced Dysarthric Speech')
plt.specgram(enhanced_dysarthric_deep_learning, Fs=sr_dysarthric, cmap='viridis')
plt.colorbar(format='%+2.0f dB')

    # Visualize the original and enhanced dysarthric speech signals (optional)
plt.subplot(3, 1, 3)
plt.title('Time-domain Signals')
plt.plot(dysarthric_speech, label='Original Dysarthric Speech')
plt.plot(enhanced_dysarthric_deep_learning, label='Enhanced Dysarthric Speech (Deep Learning)')
plt.legend()

plt.tight_layout()
plt.show()



