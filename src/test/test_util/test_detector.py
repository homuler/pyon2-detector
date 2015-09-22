import os
import numpy as np
from util import detector
from nose.tools import raises

class TestDetector:
   @classmethod
   def setup_class(clazz):
      clazz.root = 'resources/test_util'
      clazz.exp = 0.000000001
      clazz.detector = detector.Detector(os.path.join(clazz.root, 'model.dump'),
                                os.path.join(clazz.root, 'mean.npy'), gpu=-1)

   @classmethod
   def teardown_class(clazz):
      pass

   def setup(self):
      pass

   def teardown(self):
      pass

   def test_get_IoU1(self):
      """ get_IoU case1: no intersection """
      pos1 = np.array([0, 0, 50, 50], dtype=np.int32)
      pos2 = np.array([50, 50, 100, 100], dtype=np.int32)
      iou = self.detector.get_IoU(pos1, pos2)
      ans = 0
      assert abs(iou - ans) < self.exp

   def test_get_IoU2(self):
      """ get_IoU case2: totally intersect """
      pos1 = np.array([10, 10, 60, 60], dtype=np.int32)
      pos2 = np.array([10, 10, 60, 60], dtype=np.int32)
      iou = self.detector.get_IoU(pos1, pos2)
      ans = 1 
      assert abs(iou - ans) < self.exp

   def test_get_IoU3(self):
      """ get_IoU case3: x-axis shares """
      pos1 = np.array([10, 10, 60, 60], dtype=np.int32)
      pos2 = np.array([30, 40, 60, 70], dtype=np.int32)
      iou = self.detector.get_IoU(pos1, pos2)
      ans = 600 / 2800 
      assert abs(iou - ans) < self.exp

   def test_get_IoU4(self):
      """ get_IoU case4: x-axis shares """
      pos2 = np.array([10, 10, 60, 60], dtype=np.int32)
      pos1 = np.array([30, 40, 60, 70], dtype=np.int32)
      iou = self.detector.get_IoU(pos1, pos2)
      ans = 600 / 2800 
      assert abs(iou - ans) < self.exp

   def test_get_IoU5(self):
      """ get_IoU case5: rect1 contains rect2 """
      pos1 = np.array([0, 0, 60, 60], dtype=np.int32)
      pos2 = np.array([20, 20, 50, 50], dtype=np.int32)
      iou = self.detector.get_IoU(pos1, pos2)
      ans = 900 / 3600 
      assert abs(iou - ans) < self.exp

   def test_get_IoU6(self):
      """ get_IoU case6: rect2 contains rect1 """
      pos2 = np.array([0, 0, 60, 60], dtype=np.int32)
      pos1 = np.array([20, 20, 50, 50], dtype=np.int32)
      iou = self.detector.get_IoU(pos1, pos2)
      ans = 900 / 3600 
      assert abs(iou - ans) < self.exp

   def test_get_IoU7(self):
      """ get_IoU case7: shares y-axis """
      pos2 = np.array([96, 144, 144, 192], dtype=np.int32)
      pos1 = np.array([144, 108, 240, 204], dtype=np.int32)
      iou = self.detector.get_IoU(pos1, pos2)
      ans = 0
      assert abs(iou - ans) < self.exp

