import os
from nose.tools import raises
from util import loader, visualizer

class TestLoaderCpu:
   @classmethod
   def setup_class(clazz):
      clazz.root = 'resources/test_util'

   @classmethod
   def teardown_class(clazz):
      pass

   def setup(self):
      pass

   def teardown(self):
      pass

   def test_save_model_graph(self):
      pass

   def test_save_accuracy_curve1(self):
      """ save_accuracy_curve case1: simply call the method. """
      logpath = os.path.join(self.root, 'training-log.txt')
      filepath = os.path.join(self.root, 'training-log.png')
      if os.path.exists(filepath):
         os.remove(filepath)
      visualizer.save_accuracy_curve(logpath, filepath)
      assert os.path.exists(filepath)

   def test_save_loss_curve1(self):
      """ save_loss_curve case1: simply call the method. """
      logpath = os.path.join(self.root, 'training-log.txt')
      filepath = os.path.join(self.root, 'training-log.png')
      if os.path.exists(filepath):
         os.remove(filepath)
      visualizer.save_loss_curve(logpath, filepath)
      assert os.path.exists(filepath)

   def test_partition_data(self):
      pass
