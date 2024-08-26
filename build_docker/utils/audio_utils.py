import math
import random
import torch
import torchaudio
import librosa
import numpy as np
from torchaudio import transforms

class AudioUtil():

    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    @staticmethod
    def open_lr(audio_file):
        sig, sr = librosa.load(audio_file)
        return (sig, sr)

    @staticmethod
    def hpss(y):
        y_harmonic, y_percussive = librosa.effects.hpss(y = y)
        return y_harmonic, y_percussive

    @staticmethod
    def pad_trunc(sig, sr, max_ms):

        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_end_len = max_len - sig_len
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((sig, pad_end), 1)
            
        return (sig, sr)

    @staticmethod
    def spectro_gram(sig, sr, n_mels=128, n_fft=2048, hop_len=512):
        top_db = 125
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)