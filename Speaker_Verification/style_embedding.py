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

config = get_config()
tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

def to_spectrogram(utter_path):

    utter_min_len = (config.tisv_frame * config.hop + config.window) * config.sr    # lower bound of utterance length
    utterances_spec = []
    utter, sr = librosa.core.load(utter_path, config.sr)        # load utterance audio
    intervals = librosa.effects.split(utter, top_db=20)         # voice activity detection
    for interval in intervals:
        if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
            utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
            S = librosa.core.stft(y=utter_part, n_fft=config.nfft,
                                  win_length=int(config.window * sr), hop_length=int(config.hop * sr))
            S = np.abs(S) ** 2
            mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=40)
            S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances

            utterances_spec.append(S[:, :config.tisv_frame])    # first 180 frames of partial utterance
            utterances_spec.append(S[:, -config.tisv_frame:])   # last 180 frames of partial utterance

    utterances_spec = np.array(utterances_spec)
    print(utterances_spec.shape)
    return utterances_spec

def embed(path, wav_path):
    mel = to_spectrogram(wav_path)
    uttr_count = len(mel)
    
    tf.compat.v1.reset_default_graph()
    N=1
    M=1
    # draw graph
    enroll = placeholder(shape=[None, uttr_count, 40], dtype=tf.float32) # enrollment batch (time x batch x n_mel)
    verif = placeholder(shape=[None, uttr_count, 40], dtype=tf.float32)  # verification batch (time x batch x n_mel)
    batch = tf.compat.v1.concat([enroll, verif], axis=1)

    # embedding lstm (3-layer default)
    with tf.compat.v1.variable_scope("lstm"):
        lstm_cells = [LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = MultiRNNCell(lstm_cells)    # make lstm op and variables
        outputs, _ = tf.compat.v1.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize

    print("embedded size: ", embedded.shape)

    # enrollment embedded vectors (speaker model)
    enroll_embed = normalize(tf.reduce_mean(tf.reshape(embedded[:uttr_count, :], shape= [1, uttr_count, -1]), axis=1)) # tu robi średnią po wypowiedziach, czyli de facto interesuje mnie bardziej verif_embed
    # verification embedded vectors
    verif_embed = embedded[uttr_count:, :]

    saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables())
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()

        # load model
        print("model path :", path)
        ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_dir=os.path.join(path, "Check_Point"))
        ckpt_list = ckpt.all_model_checkpoint_paths
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
            result_embedding = sess.run(enroll_embed, feed_dict={enroll:mel_to_batch(mel),
                                                       verif:mel_to_batch(mel)})
        else:
            result_embedding = sess.run(enroll_embed, feed_dict={enroll:mel_to_batch(mel),
                                                       verif:mel_to_batch(mel)})
        print(result_embedding)
        return result_embedding

def mel_to_batch(utter):
    utter_batch = np.asarray(utter)
    utter_batch = utter_batch[:,:,:160]               # for test session, fixed length slicing of input batch
    print(utter_batch.shape)
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

wav_path = r'/home/maciej/Desktop/tf_working_dir/SKAJPAI/voice_style_transfer/Speaker_Verification/p232_019.wav'
            
if __name__ == "__main__":
    if os.path.isdir(config.model_path):
        embed(config.model_path, wav_path)
    else:
        raise AssertionError("model path doesn't exist!")