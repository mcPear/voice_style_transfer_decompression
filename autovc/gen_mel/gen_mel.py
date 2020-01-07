"""generate mel-spectrogram from wav"""
import librosa
from scipy import misc
import pickle
import numpy as np
import gen_mel.audio as audio
import gen_mel.hparams_gen_melspec as hparams
import os
import glob
from tqdm import tqdm

def gen_mel(wav_path):
	basename=os.path.basename(wav_path).split('.wav')[0]
	wav = audio.load_wav(wav_path)
	wav = wav / np.abs(wav).max() * hparams.hparams['rescaling_max']

	out = wav
	constant_values = 0.0
	out_dtype = np.float32

	mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
	return mel_spectrogram

