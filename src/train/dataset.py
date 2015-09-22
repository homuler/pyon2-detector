import os
import numpy as np
import six.moves.cPickle as pickle

from PIL import Image

def is_train_data(path):
   _, ext = os.path.splitext(path)
   return ext == '.png' or ext == '.jpg' or ext == '.bmp'

def create_dataset(indir, outdir, width, height):
   if not os.path.exists(outdir):
      os.makedirs(outdir)
   create_dataset_recursive(indir, outdir, width, height)

def create_dataset_recursive(indir, outdir, width, height):
   files = os.listdir(indir)
   for file in files:
      inpath = os.path.join(indir, file)
      outpath = os.path.join(outdir, file)
      if os.path.isdir(inpath):
         if not os.path.exists(outpath):
            os.makedirs(outpath)
         create_dataset_recursive(inpath, outpath, width, height)
      elif is_train_data(outpath):
         save_image(Image.open(inpath).convert("RGB"), outpath, width, height)

def save_image(image, path, width, height):
   converted = image.resize((width, height), Image.ANTIALIAS)
   converted.save(path)

def create_datalist(dirpath, filepath):
   xs = enum_dataset_recursive([], dirpath)
   f = open(filepath, 'w+')
   for path, label in xs:
      f.write(path + ' ' + label + '\n')
   f.close()

def enum_dataset_recursive(xs, dirpath, label='-1'):
   files = os.listdir(dirpath)

   for file in files:
      fullpath = os.path.join(dirpath, file)
      if os.path.isdir(fullpath):
         xs = enum_dataset_recursive(xs, fullpath, file)
      elif is_train_data(fullpath):
         xs.append((fullpath, label))

   return xs

def calc_mean(dataset):
   sum_image = None
   count = 0
   for line in open(dataset):
      tuple = line.strip().split()
      if len(tuple) == 0:
         continue
      elif len(tuple) > 2:
         raise ValueError("list file format isn't correct: [filepath] [label]")
      filepath = tuple[0]
      image = np.asarray(Image.open(filepath)).transpose(2, 0, 1)
      if sum_image is None:
         sum_image = np.ndarray(image.shape, dtype=np.float32)
         sum_image[:] = image
      else:
         sum_image += image
      count += 1
   mean = sum_image
   if not mean is None:
      mean = mean / count
   return mean 

def create_mean(dataset, filepath):
   mean = calc_mean(dataset)
   pickle.dump(mean, open(filepath, 'wb'), -1)
