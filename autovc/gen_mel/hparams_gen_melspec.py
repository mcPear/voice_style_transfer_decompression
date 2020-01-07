import tensorflow as tf
import numpy as np

hparams = {
    'name':"wavenet_vocoder",
    'builder':"wavenet",
    'input_type':"raw",
    'quantize_channels':65536,
    'sample_rate':16000,
    'silence_threshold':2,
    'num_mels':80,
    'fmin':90,
    'fmax':7600,
    'fft_size':1024,
    'hop_size':256,
    'frame_shift_ms':None,
    'min_level_db':-100,
    'ref_level_db':16,
    'rescaling':True,
    'rescaling_max':0.999,
    'allow_clipping_in_normalization':True
}

def hparams_debug_string():
    #values = hparams.values()
    #hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    #return 'Hyperparameters:\n' + '\n'.join(hp)
    return "mocked hparams string"
