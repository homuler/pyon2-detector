#!/usr/bin/env python
import numpy as np
import argparse
import os
import sys

from PIL import Image
import six.moves.cPickle as pickle

from chainer import optimizers
from network.frgnet64 import FrgNet64
from network.frgnet128 import FrgNet128

from train import batch, dataset
from util import loader

parser = argparse.ArgumentParser(description='Learner')
parser.add_argument('option', help='option \
                     (genlist, genmean, gendata, setup, learn)')
parser.add_argument('--rawdata', help='dataset directory path, that will be resized.')
parser.add_argument('--dataset', help='dataset directory path, that is used for training, validating.')
parser.add_argument('--model', help='model name')
parser.add_argument('--output', help='path, where the output will be saved.\
                                      genmean -> mean file, genlist -> list file, \
                                      setup -> test dataset.')
parser.add_argument('--size', help='picture size', type=int, default=64)
parser.add_argument('--mean', help='mean array file.')
parser.add_argument('--train', help='path to training image-label list file',
                    default='train-list.txt')
parser.add_argument('--valid', help='path to validation image-label list file',
                    default='valid-list.txt')
parser.add_argument('--gpu', type=int, default=-1, help='gpu flag')

args = parser.parse_args()

def train(trainlist, validlist, meanpath, modelname, batchsize, max_epoch=100, gpu=-1):
   model = None
   if modelname == "frg64":
      model = FrgNet64()
   elif modelname == "frg128":
      model = FrgNet128()
   optimizer = optimizers.MomentumSGD(lr=0.005, momentum=0.9)
   trainer = batch.Trainer(trainlist, validlist, meanpath, model, 
                           optimizer, 0.0001, gpu)

   trainer.train_random(batchsize, lr_decay=0.95, valid_interval=1000, 
                        model_interval=5, log_interval=20, max_epoch=max_epoch)

if args.option == 'genlist':
   dataset.create_datalist(args.dataset, args.output)
elif args.option == 'gendata':
   dataset.create_dataset(args.rawdata, args.output, args.size, args.size)
elif args.option == 'genmean':
   dataset.create_mean(args.train, args.output)
elif args.option == 'train':
   train(args.train, args.valid, args.mean, args.model, 10, max_epoch=30, gpu=args.gpu)
