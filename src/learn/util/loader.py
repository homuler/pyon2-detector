import os
import numpy as np
import six.moves.cPickle as pickle

from PIL import Image

def unpickle(filepath):
   return pickle.load(open(filepath, 'rb'))

def load_model(filepath):
   """ load trained model.
       If the model is trained on GPU, then you must initialize cuda-driver before.
   """
   return unpickle(filepath)

   
def load_mean(filepath):
   """ load mean file
   """
   return unpickle(filepath)

def load_image_list(filepath):
   """ load image-file list. Image-file list file consists of filepath and the label.
   """
   tuples = []
   for line in open(filepath):
      pair = line.strip().split()
      if len(pair) == 0:
         continue
      elif len(pair) > 2:
         raise ValueError("list file format isn't correct: [filepath] [label]")
      else:
         tuples.append((pair[0], np.int32(pair[1])))
   return tuples

def image2array(img):
   return np.asarray(img).transpose(2, 0, 1).astype(np.float32)

def load_image(path, mean, flip=False):
   image = image2array(Image.open(path))
   image -= mean
   if flip:
      return image[:, :, ::-1]
   else:
      return image

