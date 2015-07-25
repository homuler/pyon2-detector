import os
import numpy as np
from nose.tools import raises, nottest
from train import batch
from util import loader
from chainer import optimizers
from network.sample import SampleNet

class TestBatch:
   @classmethod
   def setup_class(clazz):
      clazz.root = 'resources/test_train'
      meanpath = os.path.join(clazz.root, 'mean.npy')
      trainlist = os.path.join(clazz.root, 'image-list3.txt')
      validlist = os.path.join(clazz.root, 'image-list3.txt')
      model = SampleNet() 
      optimizer = optimizers.SGD()
      clazz.trainer = batch.Trainer(trainlist, validlist, meanpath, 
                                     model, optimizer)

   @classmethod
   def teardown_class(clazz):
      pass

   def setup(self):
      pass

   def teardown(self):
      pass

   def test_create_minibatch1(self):
      """ create_minibatch case1: empty dataset """
      minibatch, labels = self.trainer.create_minibatch([])
      assert minibatch.shape == (0, 3, 64, 64)
      assert labels.shape == (0,)

   def test_create_minibatch2(self):
      """ create_minibatch case2: 2-size minibatch """
      trainer = self.trainer
      minibatch, labels = trainer.create_minibatch(trainer.trainset)
      assert minibatch.shape == (2, 3, 64, 64)
      assert labels.shape == (2,)
   
   def test_create_minibatch_random1(self):
      """ create_minibatch_random case1: empty dataset """
      minibatch, labels = self.trainer.create_minibatch_random([], 0)
      assert minibatch.shape == (0, 3, 64, 64)
      assert labels.shape == (0,)
      
   def test_create_minibatch_random2(self):
      """ create_minibatch_random case2: empty dataset (img-list is not empty) """
      trainer = self.trainer
      minibatch, labels = trainer.create_minibatch_random(trainer.trainset, 0)
      assert minibatch.shape == (0, 3, 64, 64)
      assert labels.shape == (0,)
      
   def test_create_minibatch_random3(self):
      """ create_minibatch_random case3: 2-size minibatch """
      trainer = self.trainer
      minibatch, labels = trainer.create_minibatch_random(trainer.trainset, 2)
      assert minibatch.shape == (2, 3, 64, 64)
      assert labels.shape == (2,)

   def test_create_minibatch_random4(self):
      """ create_minibatch_random case4: minibatch size is larger than img-list size """
      trainer = self.trainer
      minibatch, labels = trainer.create_minibatch_random(trainer.trainset, 3)
      assert minibatch.shape == (3, 3, 64, 64)
      assert labels.shape == (3,)
      assert not np.all(minibatch[2] == 0.0)

   def test_create_minibatch_random4(self):
      """ create_minibatch_random case4: minibatch size is smaller than img-list size """
      trainer = self.trainer
      minibatch, labels = trainer.create_minibatch_random(trainer.trainset, 1)
      assert minibatch.shape == (1, 3, 64, 64)
      assert labels.shape == (1,)

   def test_forward_minibatch1(self):
      """ forward_minibatch case1: simply call the method. """
      trainer = self.trainer
      minibatch, labels = trainer.create_minibatch(trainer.trainset)
      loss, acc = trainer.forward_minibatch(minibatch, labels)
      assert loss.data.shape == ()
      assert loss.data.size == 1
      assert acc.data.shape == ()
      assert acc.data.size == 1

   def test_forward_minibatch2(self):
      """ forward_minibatch case2: simply call the method (not train). """
      trainer = self.trainer
      minibatch, labels = trainer.create_minibatch(trainer.trainset)
      loss, acc = trainer.forward_minibatch(minibatch, labels, train=False)
      assert loss.data.shape == ()
      assert loss.data.size == 1
      assert acc.data.shape == ()
      assert acc.data.size == 1

   @raises(ValueError)
   def test_forward_data_seq1(self):
      """ forward_data_seq case1: empty dataset. """
      trainer = self.trainer
      loss, acc = trainer.forward_data_seq(trainer.trainset, 0)

   def test_forward_data_seq2(self):
      """ forward_data_seq case2: batch size is equal to dataset size. """
      trainer = self.trainer
      loss, acc = trainer.forward_data_seq(trainer.trainset, 2)
      assert loss >= 0

   def test_forward_data_seq3(self):
      """ forward_data_seq case3: batch size is larger than dataset size. """
      trainer = self.trainer
      loss, acc = trainer.forward_data_seq(trainer.trainset, 3)
      assert loss >= 0

   def test_forward_data_seq4(self):
      """ forward_data_seq case4: batch size is smaller than dataset size. """
      trainer = self.trainer
      loss, acc = trainer.forward_data_seq(trainer.trainset, 1)
      assert loss >= 0

   def test_forward_data_seq5(self):
      """ forward_data_seq case5: not in train """
      trainer = self.trainer
      loss, acc = trainer.forward_data_seq(trainer.trainset, 2, train=False)
      assert loss >= 0

   def test_train_random1(self):
      """ train_random case1: no forwarding. """
      self.trainer.train_random(0, model_interval=10, max_epoch=10)

   def test_train_random2(self):
      """ train_random case2: simply call the method. """
      self.trainer.train_random(2,  model_interval=1, log_interval=1, valid_interval=1, max_epoch=1)
