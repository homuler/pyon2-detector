import os
import shutil
import numpy as np
import six
import six.moves.cPickle as pickle
import math
from PIL import Image
from itertools import chain
import matplotlib.pyplot as plt

from chainer import cuda
from chainer import computational_graph as c
from train import dataset
from util import loader, parser
from network.manager import NetSet

def save_model_graph(loss, filepath, remove_split=False):
   with open(filepath, 'w') as o:
      o.write(c.build_computational_graph((loss,), remove_split).dump())
   
def extract_log_info(f, logdata):
   return list(map(lambda ys: list(map(f, ys)), logdata))

def save_loss_curve(logpath, filepath):
   train_log, valid_log = parser.parse_training_log(logpath)
   train_loss_log = extract_log_info(lambda x: x['loss'], train_log)
   valid_loss_log = extract_log_info(lambda x: x['loss'], valid_log)
   epoch = len(train_loss_log)
   ts = list(chain.from_iterable(train_loss_log))
   vs = list(chain.from_iterable(valid_loss_log))
   skip = len(ts) // len(vs)
   ts = ts[::skip]
   plt.figure(0)
   plt.plot(ts, label='training')
   plt.plot(vs, label='validation')
   plt.xticks([x for x in range(0, len(vs), len(vs) // 10)], 
           [x for x in range(0, epoch, epoch // 10)])
   plt.legend()
   plt.title('Loss')
   plt.xlabel('epoch')
   plt.ylabel('Loss')
   plt.savefig(filepath)
   
def save_accuracy_curve(logpath, filepath):
   train_log, valid_log = parser.parse_training_log(logpath)
   train_acc_log = extract_log_info(lambda x: x['accuracy'], train_log)
   valid_acc_log = extract_log_info(lambda x: x['accuracy'], valid_log)
   ts = list(chain.from_iterable(train_acc_log))
   vs = list(chain.from_iterable(valid_acc_log))
   epoch = len(train_acc_log)
   skip = len(ts) // len(vs)
   ts = ts[::skip]
   plt.figure(1)
   plt.plot(ts, label='training')
   plt.plot(vs, label='validation')
   plt.xticks([x for x in range(0, len(vs), len(vs) // 10)], 
           [x for x in range(0, epoch, epoch // 10)])
   plt.ylim([0.5, 1.05])
   plt.title('Accuracy')
   plt.xlabel('epoch')
   plt.ylabel('accuracy')
   plt.savefig(filepath)

def tile_sample_images_from_dir(dirpath, outpath, size, w=16, h=9):
   files = list(map(lambda x: os.path.join(dirpath, x), os.listdir(dirpath)))
   files = list(filter(lambda x: x.endswith('png'), files))
   tile_sample_images(files, outpath, size, w=w, h=h)

def tile_sample_images_from_list(listpath, outpath, size, label=-1, w=16, h=9):
   imglist = loader.load_image_list(listpath)
   imglist = list(filter(lambda x: True if(label < 0) else x[1] == label, imglist))
   imglist = list(map(lambda x: x[0], imglist))
   tile_sample_images(imglist, outpath, size, w=w, h=h)

def tile_sample_images(imglist, outpath, size, w=16, h=9):
   print('called', w, h)
   n = w * h
   xs = np.random.random_integers(0, len(imglist) - 1, n)
   canvas = Image.new('RGB', (size * w, size * h))
   print(n, xs)
   for x in range(0, h):
      for y in range(0, w):
         print(len(xs), x * h + y, len(imglist), xs[x*h + y]) 
         path = imglist[xs[x * h + y]]
         print(path)
         img = Image.open(path).resize((size, size))
         canvas.paste(img, (y * size, x * size))
   canvas.save(outpath, 'PNG')

class ModelVisualizer(NetSet):
   """ Model Visualizer """
   def __init__(self, modelpath, meanpath, gpu=-1):
      model = loader.load_model(modelpath)
      super(ModelVisualizer, self).__init__(meanpath, model, gpu)
      self.gpu = gpu
      if self.gpu >= 0:
         cuda.init(self.gpu)

   def partition_data(self, imglist, outdir, batchsize=100, confidence=1.0, gpu=-1):
      dataset = loader.load_image_list(imglist)
      correct_count = 0
      wrong_count = 0
      for i in six.moves.range(1, len(dataset), batchsize):
         mini_dataset = dataset[i:i+batchsize]
         x_batch, y_batch = self.create_minibatch(mini_dataset)
         confs = self.model.calc_confidence(x_batch)
         labels = self.calc_max_label(confs.data)
         for idx, tuple in enumerate(labels):
            label, conf = tuple
            path, ans = mini_dataset[idx]
            if label == ans:
               if conf < confidence: 
                  print('correct: path={}, confidence={}'.format(path, conf))
               save_path = os.path.join(os.path.join(outdir, 'yes'), str(label))
               if not os.path.exists(save_path):
                  os.makedirs(save_path)
               correct_count += 1
               shutil.copyfile(path, os.path.join(save_path, str(conf) + '-' + str(correct_count) + '.png'))
            else:
               print('wrong: path={}, confidence={}'.format(path, conf))
               save_path = os.path.join(os.path.join(outdir, 'no'), str(label))
               if not os.path.exists(save_path):
                  os.makedirs(save_path)
               wrong_count += 1
               shutil.copyfile(path, os.path.join(save_path, str(conf) + '-' + str(wrong_count) + '.png'))
      print('correct={}, wrong={}, accuracy rate={}'.format(
         correct_count, wrong_count, correct_count / len(dataset)))
