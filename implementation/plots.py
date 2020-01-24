import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle
from scipy.io.wavfile import write
import glob
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
#%matplotlib inline

def plot(x,x_trg,result=None):
    print(x.shape, x_trg.shape)
    plt.figure(figsize=(12, 12))
    plt.subplot(3,1,1)
    librosa.display.specshow(x, sr=16000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original utterance spectrogram')

    plt.subplot(3,1,2)
    librosa.display.specshow(x_trg, sr=16000)
    plt.colorbar(format='%+2.00f dB')
    plt.title('Compressed utterance spectrogram')

    plt.subplot(3,1,3)
    librosa.display.specshow(result, sr=16000)
    plt.colorbar(format='%+2.00f dB')
    plt.title('Outpt spectrogram')

    plt.show()