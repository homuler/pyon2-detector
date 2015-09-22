import os
import shutil
from nose.tools import raises
from train import dataset

class TestDataset:
   @classmethod
   def setup_class(clazz):
      pass

   @classmethod
   def teardown_class(clazz):
      pass

   def setup(self):
      pass

   def teardown(self):
      pass

   def test_load_model(self):
      pass

   def test_is_train_data1(self):
      """ train_data case1 : png file """
      filepath = 'test0.png'
      assert dataset.is_train_data(filepath)

   def test_is_train_data2(self):
      """ train_data case2 : png file under directory """
      filepath = 'test/test0.png'
      assert dataset.is_train_data(filepath)

   def test_is_train_data3(self):
      """ train_data case3 : jpg file """
      filepath = 'test0.jpg'
      assert dataset.is_train_data(filepath)

   def test_is_train_data4(self):
      """ train_data case4 : jpg file under directory """
      filepath = 'test/test0.png'
      assert dataset.is_train_data(filepath)

   def test_is_train_data5(self):
      """ train_data case5 : bmp file """
      filepath = 'test0.bmp'
      assert dataset.is_train_data(filepath)

   def test_is_train_data6(self):
      """ train_data case6 : bmp file under directory """
      filepath = 'test/test0.bmp'
      assert dataset.is_train_data(filepath)

   def test_is_train_data7(self):
      """ train_data case7 : txt file """
      filepath = 'test0.txt'
      assert not dataset.is_train_data(filepath)

   def test_is_train_data8(self):
      """ train_data case8 : txt file under directory """
      filepath = 'test/test0.txt'
      assert not dataset.is_train_data(filepath)

   def test_create_dataset1(self):
      """ create_dataset case1: empty dataset """
      indir = 'resources/test_train/dataset1/in'
      outdir = 'resources/test_train/dataset1/out'
      if os.path.exists(outdir):
         shutil.rmtree(outdir)
         os.mkdir(outdir)
      dataset.create_dataset(indir, outdir, 0, 0)
      assert os.listdir(os.path.join(outdir, '')) == []

   def test_create_dataset2(self):
      """ create_dataset case2: out directory exists """
      indir = 'resources/test_train/dataset2/in'
      outdir = 'resources/test_train/dataset2/out'
      if os.path.exists(outdir):
         shutil.rmtree(outdir)
         os.mkdir(outdir)
      else:
         os.mkdir(outdir)
      dataset.create_dataset(indir, outdir, 64, 64)
      dir0 = os.path.join(outdir, '0')
      dir1 = os.path.join(outdir, '1')

      xs = os.walk(outdir)
      files = []
      for path, ds, fs in xs:
         if len(fs) > 0:
            for f in fs:
               files.append(os.path.join(path, f))
      res = [os.path.join(dir0, 'test1.png'), os.path.join(dir1, 'test2.png')]
      assert sorted(files) == sorted(res)

   def test_create_dataset3(self):
      """ create_dataset case3: out directory not exists """
      indir = 'resources/test_train/dataset3/in'
      outdir = 'resources/test_train/dataset3/out'
      if os.path.exists(outdir):
         shutil.rmtree(outdir)
         os.mkdir(outdir)
      dataset.create_dataset(indir, outdir, 64, 64)
      files = os.listdir(outdir)
      res = ['test1.png', 'test2.png']
      assert sorted(files) == sorted(res)

   def test_create_dataset4(self):
      """ create_dataset case4: dataset not exists """
      indir = 'resources/test_train/dataset4/in'
      outdir = 'resources/test_train/dataset4/out'
      if os.path.exists(outdir):
         shutil.rmtree(outdir)
         os.mkdir(outdir)
      dataset.create_dataset(indir, outdir, 64, 64)
      assert os.listdir(os.path.join(outdir, '')) == []

   def test_enum_dataset_recursive1(self):
      """ enum_dataset_recursive case1: dataset not exists """
      dirpath = 'resources/test_train/dataset1/in'
      files = dataset.enum_dataset_recursive([], dirpath)
      assert files == []

   def test_enum_dataset_recursive2(self):
      """ enum_dataset_recursive case2: search recursively """
      dirpath = 'resources/test_train/dataset2/in'
      files = dataset.enum_dataset_recursive([], dirpath)
      dir0 = os.path.join(dirpath, '0')
      dir1 = os.path.join(dirpath, '1')
      res = [(os.path.join(dir0, 'test1.png'), '0'), (os.path.join(dir1, 'test2.png'), '1')]
      assert sorted(files) == sorted(res)

   def test_enum_dataset_recursive3(self):
      """ enum_dataset_recursive case3: search not recursively (has no label)"""
      dirpath = 'resources/test_train/dataset3/in'
      files = dataset.enum_dataset_recursive([], dirpath)
      res = [(os.path.join(dirpath, 'test1.png'), '-1'), (os.path.join(dirpath, 'test2.png'), '-1')]
      assert sorted(files) == sorted(res)

   def test_enum_dataset_recursive4(self):
      """ enum_dataset_recursive case4: not exists train data """
      dirpath = 'resources/test_train/dataset4/in'
      files = dataset.enum_dataset_recursive([], dirpath)
      assert files == []

   def test_create_datalist1(self):
      """ create_datalist case1: empty dataset """
      dirpath = 'resources/test_train/dataset1/in'
      filepath = 'resources/test_train/dataset1/out/image-list.txt'
      dataset.create_datalist(dirpath, filepath)
      assert os.path.exists(filepath)

   def test_create_datalist2(self):
      """ create_datalist case2: search dataset recursively """
      dirpath = 'resources/test_train/dataset2/in'
      filepath = 'resources/test_train/dataset2/out/image-list.txt'
      dataset.create_datalist(dirpath, filepath)
      assert os.path.exists(filepath)

   def test_create_datalist3(self):
      """ create_datalist case3: search dataset not recursively """
      dirpath = 'resources/test_train/dataset3/in'
      filepath = 'resources/test_train/dataset3/out/image-list.txt'
      dataset.create_datalist(dirpath, filepath)
      assert os.path.exists(filepath)

   def test_create_datalist4(self):
      """ create_datalist case4: empty dataset (only text file) """
      dirpath = 'resources/test_train/dataset4/in'
      filepath = 'resources/test_train/dataset4/out/image-list.txt'
      dataset.create_datalist(dirpath, filepath)
      assert os.path.exists(filepath)

   def test_calc_mean1(self):
      """ calc_mean case1: empty dataset """
      filepath = 'resources/test_train/image-list1.txt'
      mean = dataset.calc_mean(filepath)
      assert mean is None

   def test_calc_mean2(self):
      """ calc_mean case2: 2 dataset """
      filepath = 'resources/test_train/image-list2.txt'
      mean = dataset.calc_mean(filepath)
      assert not mean is None

   def test_calc_mean3(self):
      """ calc_mean case3: datalist has an empty line. """
      filepath = 'resources/test_train/image-list3.txt'
      mean = dataset.calc_mean(filepath)
      assert not mean is None

   @raises(ValueError)
   def test_calc_mean4(self):
      """ calc_mean case4: datalist format is incorrect. """
      filepath = 'resources/test_train/image-list4.txt'
      mean = dataset.calc_mean(filepath)

   def test_create_mean1(self):
      """ calc_mean case1: empty dataset """
      filepath = 'resources/test_train/image-list1.txt'
      meanpath = 'resources/test_train/mean.npy'
      if os.path.exists(meanpath):
         os.remove(meanpath)
      dataset.create_mean(filepath, meanpath)
      assert os.path.exists(meanpath)
      
   def test_create_mean2(self):
      """ calc_mean case2: 2 dataset """
      filepath = 'resources/test_train/image-list2.txt'
      meanpath = 'resources/test_train/mean.npy'
      if os.path.exists(meanpath):
         os.remove(meanpath)
      dataset.create_mean(filepath, meanpath)
      assert os.path.exists(meanpath)
