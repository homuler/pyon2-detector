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
from util import loader, detector, visualizer

def genlist(args):
   dataset.create_datalist(args.dataset, args.output)

def gendata(args):
   dataset.create_dataset(args.rawdata, args.output, args.size, args.size)

def genmean(args):
   dataset.create_mean(args.train, args.output)

def learn(args):
   batchsize = 20
   max_epoch = 30
   model = None
   print('model initializing...')
   if args.model == "frg64":
      model = FrgNet64()
   elif args.model == "frg128":
      model = FrgNet128()
   optimizer = optimizers.MomentumSGD(lr=0.005, momentum=0.9)
   print('trainer initializing...')
   trainer = batch.Trainer(args.train, args.valid, args.mean, model, 
                           optimizer, 0.0001, args.gpu)

   trainer.train_random(batchsize, lr_decay=0.98, valid_interval=1000, 
                        model_interval=5, log_interval=20, max_epoch=max_epoch)

def detect(args):
    d = detector.Detector(args.model, args.mean, args.gpu)
    d.sliding_window(args.input, [(48, 16), (72, 24), (144, 48)], output=args.output, confidence=0.9)

def visualizeLog(args):
    visualizer.save_accuracy_curve(args.input, args.acc)
    visualizer.save_loss_curve(args.input, args.loss)

parser = argparse.ArgumentParser(description='Pyon2-Detector')
subparsers = parser.add_subparsers()

genlistParser = subparsers.add_parser('genlist')
genlistParser.add_argument('--dataset', help='dataset directory path, that is used for training, validating.')
genlistParser.add_argument('--output', help='path, where the list file will be saved.')
genlistParser.set_defaults(func=genlist)

gendataParser = subparsers.add_parser('gendata')
gendataParser.add_argument('--size', help='picture size', type=int, default=64)
gendataParser.add_argument('--rawdata', help='dataset directory path, that will be resized.')
gendataParser.add_argument('--output', help='directory path, where the data-set will be saved.')
gendataParser.set_defaults(func=gendata)

genmeanParser = subparsers.add_parser('genmean')
genmeanParser.add_argument('--train', help='path to training image-label list file')
genmeanParser.add_argument('--output', help='path, where the mean file will be saved.')
genmeanParser.set_defaults(func=genmean)

learnParser = subparsers.add_parser('learn')
learnParser.add_argument('--train', help='path to training image-label list file',
                    default='train-list.txt')
learnParser.add_argument('--valid', help='path to validation image-label list file',
                    default='valid-list.txt')
learnParser.add_argument('--gpu', type=int, default=-1, help='gpu flag')
learnParser.add_argument('--mean', help='mean array file.')
learnParser.add_argument('--model', help='model name')
learnParser.set_defaults(func=learn)

visualizeParser = subparsers.add_parser('visualize')
visualizeParser.add_argument('--input', help='path to a log file')
visualizeParser.add_argument('--acc', default='accuracy.png', help='path to accuracy data image')
visualizeParser.add_argument('--loss', default='loss.png', help='path to loss data image')
visualizeParser.set_defaults(func=visualizeLog)

detectParser = subparsers.add_parser('detect')
detectParser.add_argument('--gpu', type=int, default=-1, help='gpu flag')
detectParser.add_argument('--mean', help='mean array file.')
detectParser.add_argument('--model', help='model name')
detectParser.add_argument('--input', help='path to a image file, from which detector detects faces')
detectParser.add_argument('--output', default='test-detection.png', help='path to a image file, to which detector saves the result')
detectParser.set_defaults(func=detect)

args = parser.parse_args()
args.func(args)
