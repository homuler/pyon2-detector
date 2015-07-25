import os
from nose.tools import raises
from util import loader

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

   def test_load_model1(self):
      """ load_model case1: simply call the method. """
      filepath = os.path.join(self.root, 'model.dump')
      loader.load_model(filepath)

   def test_load_mean1(self):
      """ load_mean case1: simply call the method. """
      filepath = 'resources/test_util/mean.npy'
      loader.load_mean(filepath)

   def test_load_image_list1(self):
      """ load_image_list case1: image list contains no lines. """
      filepath = 'resources/test_util/image-list1.txt'
      xs = loader.load_image_list(filepath)
      assert xs == []

   def test_load_image_list2(self):
      """ load_image_list case2: image list contains 1 line. """
      filepath = 'resources/test_util/image-list2.txt'
      xs = loader.load_image_list(filepath)
      assert xs == [('test1.png', 0)]

   def test_load_image_list3(self):
      """ load_image_list case3: image list contains more than 2 lines, \
and an empty line. """
      filepath = 'resources/test_util/image-list3.txt'
      xs = loader.load_image_list(filepath)
      assert xs == [('test1.png', 0), ('test2.png', 1)]

   @raises(ValueError)
   def test_load_image_list4(self):
      """ load_image_list case4: image list contains a line \
that has more than 2 values. """
      filepath = 'resources/test_util/image-list4.txt'
      xs = loader.load_image_list(filepath)
      
   def test_load_image1(self):
      imgpath = 'resources/test_util/dataset/test1.png'
      meanpath = 'resources/test_util/mean.npy'
      mean = loader.load_mean(meanpath)
      loader.load_image(imgpath, mean)

   def test_load_image2(self):
      imgpath = 'resources/test_util/dataset/test2.png'
      meanpath = 'resources/test_util/mean.npy'
      mean = loader.load_mean(meanpath)
      loader.load_image(imgpath, mean, True)
