import numpy as np
import sys
import time
import six
import six.moves.cPickle as pickle
from util import loader, visualizer
from chainer import cuda, optimizers
from network.manager import NetSet

class Trainer(NetSet):
   """ train deep-network utility class """
   def __init__(self, trainlist, validlist, meanpath, model, 
                optimizer, weight_decay=0.0001, gpu=-1):
      super(Trainer, self).__init__(meanpath, model, gpu)
      self.trainset = loader.load_image_list(trainlist)
      self.validset = loader.load_image_list(validlist)
      self.optimizer = optimizer
      self.wd_rate = weight_decay
      optimizer.setup(model)
      print('set up')

   def train_random(self, batchsize, lr_decay=0.1, valid_interval=500, 
                    model_interval=10, log_interval=100, max_epoch=100):
      epoch_iter = 0
      if batchsize > 0:
         epoch_iter = len(self.trainset) // batchsize + 1
      begin_at = time.time()
      for epoch in six.moves.range(1, max_epoch + 1):
         print('epoch {} starts.'.format(epoch))
         train_duration = 0
         sum_loss = 0
         sum_accuracy = 0
         N = batchsize * log_interval
         for iter in six.moves.range(1, epoch_iter):
            iter_begin_at = time.time()
            x_batch, y_batch = self.create_minibatch_random(self.trainset, batchsize)
            loss, acc = self.forward_minibatch(x_batch, y_batch)
            train_duration += time.time() - iter_begin_at
            if epoch == 1 and iter == 1:
               visualizer.save_model_graph(loss, 'graph.dot')
               visualizer.save_model_graph(loss, 'graph.split.dot', remove_split=True)
               print('model graph is generated.')
            sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
            sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize
 
            if iter % log_interval == 0:
               throughput = batchsize * iter / train_duration
               print('training: iteration={:d}, mean loss={:.8f}, accuracy rate={:.6f}, learning rate={:f}, weight decay={:f}'
                  .format(iter + (epoch - 1) * epoch_iter, sum_loss / N, sum_accuracy / N, self.optimizer.lr, self.wd_rate))
               print('epoch {}: passed time={}, throughput ({} images/sec)'
                  .format(epoch, train_duration, throughput))
               sum_loss = 0
               sum_accuracy = 0

            if iter % valid_interval == 0:
               N_test = len(self.validset)
               valid_begin_at = time.time()
               valid_sum_loss, valid_sum_accuracy = self.forward_data_seq(self.validset, batchsize, train=False)
               valid_duration = time.time() - valid_begin_at
               throughput = N_test / valid_duration
               print('validation: iteration={:d}, mean loss={:.8f}, accuracy rate={:.6f}'
                  .format(iter + (epoch - 1) * epoch_iter, valid_sum_loss / N_test, valid_sum_accuracy / N_test))
               print('validation time={}, throughput ({} images/sec)'
                  .format(valid_duration, throughput))

            sys.stdout.flush()
         self.optimizer.lr *= lr_decay
         self.wd_rate *= lr_decay
         if epoch % model_interval == 0:
            print('saving model...(epoch {})'.format(epoch))
            pickle.dump(self.model, open('model-' + str(epoch) + '.dump', 'wb'), -1)
      print('train finished, total duration={} sec.'
         .format(time.time() - begin_at))
      pickle.dump(self.model, open('model.dump', 'wb'), -1)

   def forward_data_seq(self, dataset, batchsize, train=True):
      sum_loss = 0
      sum_accuracy = 0
      for i in range(0, len(dataset), batchsize):
         mini_dataset = dataset[i:i+batchsize]
         x_batch, y_batch = self.create_minibatch(mini_dataset)
         loss, acc = self.forward_minibatch(x_batch, y_batch, train)
         loss_data = loss.data
         acc_data = acc.data
         if self.gpu >= 0:
            loss_data = cuda.to_cpu(loss_data)
            acc_data = cuda.to_cpu(acc_data)
         sum_loss += float(loss_data) * len(mini_dataset)
         sum_accuracy += float(acc_data) * len(mini_dataset)
      return sum_loss, sum_accuracy
      
   def forward_minibatch(self, x_batch, y_batch, train=True):
      if self.gpu >= 0:
         x_batch = cuda.to_gpu(x_batch)
         y_batch = cuda.to_gpu(y_batch)

      if train:
         self.optimizer.zero_grads()

      loss, acc = self.model.forward(x_batch, y_batch, train)

      if train:
         loss.backward()
         self.optimizer.weight_decay(self.wd_rate)            
         self.optimizer.update()
      return loss, acc
