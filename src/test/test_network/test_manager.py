import os
import numpy as np
from nose.tools import raises, nottest
from train import batch
from util import loader
from chainer import optimizers
from network.frgnet64 import FrgNet64
from network.manager import NetSet

class TestNetSet:
   @classmethod
   def setup_class(clazz):
      clazz.root = 'resources/test_train'
      imglist = os.path.join(clazz.root, 'image-list3.txt')
      clazz.dataset = loader.load_image_list(imglist)
      meanpath = os.path.join(clazz.root, 'mean.npy')
      model = FrgNet64() 
      clazz.hypernet = NetSet(meanpath, model)

   @classmethod
   def teardown_class(clazz):
      pass

   def setup(self):
      pass

   def teardown(self):
      pass

   def test_create_minibatch1(self):
      """ create_minibatch case1: empty dataset """
      minibatch, labels = self.hypernet.create_minibatch([])
      assert minibatch.shape == (0, 3, 64, 64)
      assert labels.shape == (0,)

   def test_create_minibatch2(self):
      """ create_minibatch case2: 2-size minibatch """
      hypernet = self.hypernet
      minibatch, labels = hypernet.create_minibatch(self.dataset)
      assert minibatch.shape == (2, 3, 64, 64)
      assert labels.shape == (2,)
   
   def test_create_minibatch_random1(self):
      """ create_minibatch_random case1: empty dataset """
      minibatch, labels = self.hypernet.create_minibatch_random([], 0)
      assert minibatch.shape == (0, 3, 64, 64)
      assert labels.shape == (0,)
      
   def test_create_minibatch_random2(self):
      """ create_minibatch_random case2: empty dataset (img-list is not empty) """
      hypernet = self.hypernet
      minibatch, labels = hypernet.create_minibatch_random(self.dataset, 0)
      assert minibatch.shape == (0, 3, 64, 64)
      assert labels.shape == (0,)
      
   def test_create_minibatch_random3(self):
      """ create_minibatch_random case3: 2-size minibatch """
      hypernet = self.hypernet
      minibatch, labels = hypernet.create_minibatch_random(self.dataset, 2)
      assert minibatch.shape == (2, 3, 64, 64)
      assert labels.shape == (2,)

   def test_create_minibatch_random4(self):
      """ create_minibatch_random case4: minibatch size is larger than img-list size """
      hypernet = self.hypernet
      minibatch, labels = hypernet.create_minibatch_random(self.dataset, 3)
      assert minibatch.shape == (3, 3, 64, 64)
      assert labels.shape == (3,)
      assert not np.all(minibatch[2] == 0.0)

   def test_create_minibatch_random4(self):
      """ create_minibatch_random case4: minibatch size is smaller than img-list size """
      hypernet = self.hypernet
      minibatch, labels = hypernet.create_minibatch_random(self.dataset, 1)
      assert minibatch.shape == (1, 3, 64, 64)
      assert labels.shape == (1,)

   def test_forward_minibatch1(self):
      """ forward_minibatch case1: simply call the method. """
      hypernet = self.hypernet
      minibatch, labels = hypernet.create_minibatch(self.dataset)
      loss, acc = hypernet.forward_minibatch(minibatch, labels)
      assert loss.data.shape == ()
      assert loss.data.size == 1
      assert acc.data.shape == ()
      assert acc.data.size == 1

   def test_forward_minibatch2(self):
      """ forward_minibatch case2: simply call the method (not train). """
      hypernet = self.hypernet
      minibatch, labels = hypernet.create_minibatch(self.dataset)
      loss, acc = hypernet.forward_minibatch(minibatch, labels, train=False)
      assert loss.data.shape == ()
      assert loss.data.size == 1
      assert acc.data.shape == ()
      assert acc.data.size == 1

   @raises(ValueError)
   def test_forward_data_seq1(self):
      """ forward_data_seq case1: empty dataset. """
      hypernet = self.hypernet
      loss, acc = hypernet.forward_data_seq(self.dataset, 0)

   def test_forward_data_seq2(self):
      """ forward_data_seq case2: batch size is equal to dataset size. """
      hypernet = self.hypernet
      loss, acc = hypernet.forward_data_seq(self.dataset, 2)
      assert loss > 0

   def test_forward_data_seq3(self):
      """ forward_data_seq case3: batch size is larger than dataset size. """
      hypernet = self.hypernet
      loss, acc = hypernet.forward_data_seq(self.dataset, 3)
      assert loss > 0

   def test_forward_data_seq4(self):
      """ forward_data_seq case4: batch size is smaller than dataset size. """
      hypernet = self.hypernet
      loss, acc = hypernet.forward_data_seq(self.dataset, 1)
      assert loss > 0
