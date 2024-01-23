import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import tensorflow as tf
from tensorflow import keras
from import layers


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
        layers.Dense(input_shape[0], activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    # Load the dysarthric speech signal
    dysarthric_speech, sr_dysarthric = sf.read(r'D:\Aneeta_Phd\Speech Datasets\English\With Dysarthria\M\M01\Session1\wav_arrayMic\0001.wav')

    # Obtain the noise segment from the dysarthric speech
    noise_segments = detect_silence(dysarthric_speech, sr_dysarthric)
    noise = np.concatenate([dysarthric_speech[start:end] for start, end in noise_segments])

    # Perform spectral subtraction for dysarthric speech enhancement
    alpha = 2.0  # Adjust this parameter to control the amount of noise reduction
    enhanced_dysarthric = spectral_subtraction(dysarthric_speech, noise, alpha)

    # Additional pre-processing for deep learning-based approach
    # Normalize the signals to the range [-1, 1]
    dysarthric_speech = dysarthric_speech / np.max(np.abs(dysarthric_speech))
    enhanced_dysarthric = enhanced_dysarthric / np.max(np.abs(enhanced_dysarthric))

    # Create the denoising autoencoder
    input_shape = (dysarthric_speech.shape[0], 1)  # Modify input_shape to have two dimensions
    model = create_denoising_autoencoder(input_shape)

    # Train the denoising autoencoder using the noisy speech as input and the clean speech as target
    model.fit(dysarthric_speech.reshape(-1, 1), dysarthric_speech.reshape(-1, 1), epochs=2, batch_size=10, validation_split=0.2)

    # Use the trained denoising autoencoder to enhance the dysarthric speech
    enhanced_dysarthric_deep_learning = model.predict(enhanced_dysarthric.reshape(-1, 1))  # Reshape the input


    # Visualize the original and enhanced dysarthric speech signals (optional)
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.title('Original Dysarthric Speech')
    plt.plot(dysarthric_speech)
    plt.subplot(2, 1, 2)
    plt.title('Enhanced Dysarthric Speech (Deep Learning)')
    plt.plot(enhanced_dysarthric_deep_learning)
    plt.tight_layout()
    plt.show()
