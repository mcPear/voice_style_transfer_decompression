import tensorflow as tf
import os
from model import train, test
from configuration import get_config
import errno    
import os
import librosa
import argparse
import numpy as np
import time
from utils import random_batch, normalize, similarity, loss_cal, optim
from configuration import get_config
from tensorflow_addons import rnn
from tensorflow.compat.v1 import placeholder
from tensorflow.compat.v1.nn.rnn_cell import LSTMCell, MultiRNNCell
import random

#RTX float16 setup:
# tf.compat.v1.keras.backend.set_floatx('float16')
# tf.compat.v1.keras.backend.set_epsilon(1e-4)

config = get_config()
tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

def to_mel_split(utter,sr):

    utter_min_len = (config.tisv_frame * config.hop + config.window) * config.sr    # lower bound of utterance length
    utterances_spec = []
    intervals = librosa.effects.split(utter, top_db=20)         # voice activity detection
    print('intervals: ',intervals)
    for interval in intervals:
        #librosa.output.write_wav(str(random.random())+'.wav', utter[interval[0]:interval[1]], sr) #debug
        #if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
        utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
        S = to_mel(utter_part,sr)

        utterances_spec.append(S[:, :config.tisv_frame])    # first 180 frames of partial utterance
        utterances_spec.append(S[:, -config.tisv_frame:])   # last 180 frames of partial utterance

    utterances_spec = np.array(utterances_spec)
    print("spectrogram: ",utterances_spec.shape)
    return utterances_spec

def to_mel(utter,sr):
    S = librosa.core.stft(y=utter, n_fft=config.nfft,
                          win_length=int(config.window * sr), hop_length=int(config.hop * sr))
    S = np.abs(S) ** 2
    mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=config.mel_size)
    S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances

    return S

def embed(wav_path, name):
    path = config.model_path
    utter, sr = librosa.core.load(wav_path, config.sr)
    mel_split = to_mel_split(utter, sr)
    uttr_count = len(mel_split)
    mel = to_mel(utter, sr)
    
    tf.compat.v1.reset_default_graph()

    # draw graph
    batch = placeholder(shape=[None, uttr_count, config.mel_size], dtype=tf.float32) # enrollment batch (time x batch x n_mel)

    # embedding lstm (3-layer default)
    with tf.compat.v1.variable_scope("lstm"):
        lstm_cells = [LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = MultiRNNCell(lstm_cells)    # make lstm op and variables
        outputs, _ = tf.compat.v1.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize

    print("embedded size: ", embedded.shape)

    # mean of embedded vectors (speaker model)
    enroll_embed = normalize(tf.reduce_mean(tf.reshape(embedded[:uttr_count, :], shape= [1, uttr_count, -1]), axis=1))

    saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables())
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()

        # load model
        print("model path :", path)
        ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_dir=os.path.join(path, "Check_Point"))
        ckpt_list = ckpt.all_model_checkpoint_paths
        print(ckpt_list)
        loaded = 0
        for model in ckpt_list:
            if config.model_num == int(model[-1]):    # find ckpt file which matches configuration model number
                print("ckpt file is loaded !", model)
                loaded = 1
                saver.restore(sess, model)  # restore variables from selected ckpt file
                break

        if loaded == 0:
            raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")

        print("test file path : ", config.test_path)

        # return similarity matrix after enrollment and verification
        time1 = time.time() # for check inference time
        if config.tdsv:
            result_embedding = sess.run(enroll_embed, feed_dict={batch:mel_to_batch(mel_split)})
        else:
            result_embedding = sess.run(enroll_embed, feed_dict={batch:mel_to_batch(mel_split)})
        print(result_embedding)
        save(result_embedding, mel, name)

def save(emb, mel, name):
    dir_path = "/tf/notebooks/SKAJPAI/voice_style_transfer/cross_model_resources/"
    np.save(dir_path+"emb_"+name, emb)
    np.save(dir_path+"mel_"+name, mel)
    
    
    
def mel_to_batch(utter):
    utter_batch = np.asarray(utter)
    print('utter_batch.shape: ',utter_batch.shape) #debug
    utter_batch = np.transpose(utter_batch, axes=(2,0,1))     # transpose [frames, batch, n_mels]
    
    return utter_batch
        
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# if __name__ == "__main__":
#     if os.path.isdir(config.model_path):
#         embed(config.model_path, wav_path, "main")
#     else:
#         raise AssertionError("model path doesn't exist!")