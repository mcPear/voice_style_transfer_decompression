# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''
from __future__ import print_function, division

#from hyperparams import Hyperparams as hp
import numpy as np
#import tensorflow as tf
import librosa
import copy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy import signal
import os
from preprocess.tacotron.gen_mel import gen_mel
#exec("/home/maciej/Desktop/tf_working_dir/SKAJPAI/voice_style_transfer/voice_conversion/preprocess/tacotron/gen_mel.py")

class hyperparams(object):
    def __init__(self):
        self.max_duration = 10.0

        # signal processing
        self.sr = 16000 # Sample rate.
        self.n_fft = 1024 # fft points (samples)
        self.frame_shift = 0.0125 # seconds
        self.frame_length = 0.05 # seconds
        self.hop_length = int(self.sr*self.frame_shift) # samples.
        self.win_length = int(self.sr*self.frame_length) # samples.
        self.n_mels = 80 # Number of Mel banks to generate
        self.power = 1.2 # Exponent for amplifying the predicted magnitude
        self.n_iter = 300 # Number of inversion iterations
        self.preemphasis = .97 # or None
        self.max_db = 100
        self.ref_db = 20

hp = hyperparams()

def get_spectrogram(fpath, wavenet_mel=False):
    
    if wavenet_mel:
        mel=gen_mel(fpath)
        print(mel.shape)
        print(mel)
        return mel.astype(np.float32) #todo test it
    
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      #mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''
    # num = np.random.randn()
    # if num < .2:
    #     y, sr = librosa.load(fpath, sr=hp.sr)
    # else:
    #     if num < .4:
    #         tempo = 1.1
    #     elif num < .6:
    #         tempo = 1.2
    #     elif num < .8:
    #         tempo = 0.9
    #     else:
    #         tempo = 0.8
    #     cmd = "ffmpeg -i {} -y ar {} -hide_banner -loglevel panic -ac 1 -filter:a atempo={} -vn temp.wav".format(fpath, hp.sr, tempo)
    #     os.system(cmd)
    #     y, sr = librosa.load('temp.wav', sr=hp.sr)

    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)


    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    #mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    #mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    #mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    #mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    #mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)
    print(mag.shape)
    print(mag)
    return mag


def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")
