import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle
from utils import Hps
from utils import DataLoader
from utils import Logger
from utils import SingleDataset
from solver import Solver
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', default=False, action='store_true')
    parser.add_argument('-flag', default='train')
    parser.add_argument('-hps_path', default="/tf/notebooks/SKAJPAI/voice_style_transfer/voice_conversion/vctk.json")
    parser.add_argument('-load_model_path')
    parser.add_argument('-dataset_path_trg', default="/tf/notebooks/SKAJPAI/voice_style_transfer/voice_conversion/vctk_old/data.h5")
    parser.add_argument('-dataset_path', default="/tf/notebooks/SKAJPAI/voice_style_transfer/voice_conversion/vctk_old/data-8.h5")
    parser.add_argument('-index_path_trg', default="/tf/notebooks/SKAJPAI/voice_style_transfer/voice_conversion/vctk_old/index.json")
    parser.add_argument('-index_path', default="/tf/notebooks/SKAJPAI/voice_style_transfer/voice_conversion/vctk_old/index-8.json")
    parser.add_argument('-output_model_path', default="/tf/notebooks/SKAJPAI/voice_style_transfer/voice_conversion/model_vctk/model_vctk")
    args = parser.parse_args()
    hps = Hps()
    hps.load(args.hps_path)
    hps_tuple = hps.get_tuple()
    dataset = SingleDataset(args.dataset_path, args.index_path, seg_len=hps_tuple.seg_len)
    dataset_trg = SingleDataset(args.dataset_path_trg, args.index_path_trg, seg_len=hps_tuple.seg_len)
    data_loader = DataLoader(dataset, dataset_trg)

    solver = Solver(hps_tuple, data_loader)
    if args.load_model:
        solver.load_model(args.load_model_path)

    solver.train(args.output_model_path, args.flag, mode='pretrain_G')
    solver.train(args.output_model_path, args.flag, mode='pretrain_D')
    solver.train(args.output_model_path, args.flag, mode='train')
    solver.train(args.output_model_path, args.flag, mode='patchGAN')
