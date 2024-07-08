import tensorflow as tf
import numpy as np
import librosa
import sys
import os
from platform import python_version
import tensorflow_io as tfio

model = tf.keras.models.load_model('voice_model.tf') 

padd = 48000

def load_wav_16k_mono(path):
    file_contents = tf.io.read_file(path)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def predict(file_path):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:padd]
    zero_padding = tf.zeros([padd] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectogram = tf.signal.stft(wav, frame_length=320, frame_step=45)
    spectogram = tf.abs(spectogram)
    spectogram = tf.expand_dims(spectogram, axis=2)
    spectogram = tf.expand_dims(spectogram, axis=0)
    prediction = model(spectogram)
    return prediction.numpy().argmax()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run.py <path_to_wav_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)

    print(predict(file_path))
