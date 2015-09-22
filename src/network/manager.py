import numpy as np
import time
import six

from util import loader
from chainer import cuda, optimizers

class NetSet:
   def __init__(self, meanpath, model, gpu=-1):
      self.mean = loader.load_mean(meanpath)
      self.model = model
      self.gpu = gpu
      self.insize = model.insize
      if gpu >= 0:
         cuda.check_cuda_available()
         cuda.get_device(gpu).use()
         self.model.to_gpu()

   def calc_max_label(self, prob_arr):
      h, w = prob_arr.shape
      labels = [0] * h
      for i in six.moves.range(0, h):
         label = prob_arr[i].argmax()
         labels[i] = (label, prob_arr[i][label])
      return labels

   def forward_data_seq(self, dataset, batchsize):
      sum_loss = 0
      sum_accuracy = 0
      for i in range(0, len(dataset), batchsize):
         mini_dataset = dataset[i:i+batchsize]
         x_batch, y_batch = self.create_minibatch(mini_dataset)
         loss, acc = self.forward_minibatch(x_batch, y_batch)
         loss_data = loss.data
         acc_data = acc.data
         if self.gpu >= 0:
            loss_data = cuda.to_cpu(loss_data)
            acc_data = cuda.to_cpu(acc_data)
         sum_loss += float(loss_data) * len(mini_dataset)
         sum_accuracy += float(acc_data) * len(mini_dataset)
      return sum_loss, sum_accuracy

   def forward_minibatch(self, x_batch, y_batch, train=False):
      if self.gpu >= 0:
         x_batch = cuda.to_gpu(x_batch)
         y_batch = cuda.to_gpu(y_batch)
      return self.model.forward(x_batch, y_batch, train=False)

   def create_minibatch(self, dataset):
      minibatch = np.ndarray(
         (len(dataset), 3, self.insize, self.insize), dtype=np.float32)
      minibatch_label = np.ndarray((len(dataset),), dtype=np.int32)
      for idx, tuple in enumerate(dataset):
         path, label = tuple
         minibatch[idx] = loader.load_image(path, self.mean, False)
         minibatch_label[idx] = label
      return minibatch, minibatch_label

   def create_minibatch_random(self, dataset, batchsize):
      """ create minibatch randomly from dataset
          :param dataset: image path and the label
          :type dataset: (string, string)
          :param batchsize: minibatch size
          :type batchsize: int
      """
      if dataset is None or len(dataset) == 0:
         return self.create_minibatch([])
      rs = np.random.random_integers(0, high=len(dataset) - 1, size=(batchsize,))
      minidataset = []
      for idx in rs:
         minidataset.append(dataset[idx])
      return self.create_minibatch(minidataset)
