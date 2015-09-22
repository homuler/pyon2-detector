import numpy as np
import six
from PIL import Image, ImageDraw
import cv2

from chainer import cuda

from network.manager import NetSet
from util import loader

class Detector(NetSet):
   def __init__(self, modelpath, meanpath, gpu=-1):
      model = loader.load_model(modelpath)
      super(Detector, self).__init__(meanpath, model, gpu)

   def regress(img, x1, y1, size):
      pass

   def get_IoU(self, pos1, pos2):
      x1, y1, x2, y2 = pos1
      x3, y3, x4, y4 = pos2
      if x2 <= x3 or x4 <= x1:
         return 0.0
      if y2 <= y3 or y4 <= y1:
         return 0.0
      ltx = max(x1, x3)
      lty = max(y1, y3)
      rbx = min(x2, x4)
      rby = min(y2, y4)
      
      rect1 = (x2 - x1) * (y2 - y1)
      rect2 = (x4 - x3) * (y4 - y3)
      union = (rbx - ltx) * (rby - lty)
      return union / (rect1 + rect2 - union)
      
   def filter_with_IoU(self, labels, pos, threshold=0.2):
      flag = [True] * len(labels)
      count = 0
      for i in range(0, len(labels)):
         l1, conf1 = labels[i]
         for j in range(i+1, len(labels)):
            l2, conf2 = labels[j]
            if i >= j or not l1 == l2:
               continue
            iou = self.get_IoU(pos[i], pos[j])
            if iou > threshold:
               if conf1 > conf2:
                  flag[j] = False
            #      print(pos[j], 'exclude', pos[i], conf1, conf2, iou)
               else:
                  flag[i] = False
            #      print(pos[i], 'exclude', pos[j], conf1, conf2, iou)
         if flag[i]:
            count += 1
      f_labels = []
      f_pos = np.ndarray((count, 4), np.int32)
      i = 0
      for idx, f in enumerate(flag):
         if not f:
            continue
         f_labels.append(labels[idx])
         f_pos[i] = pos[idx]
         i += 1
      return f_labels, f_pos
               
   def calc_conf_of_subimages(self, img, poslist, batchsize=100):
      insize = self.model.insize
      confs = None
      # print(poslist)
      for i in range(0, (len(poslist) + batchsize - 1) // batchsize):
         minibatch_size = min(batchsize, len(poslist) - i*batchsize)
         # print('batch size =', minibatch_size)
         minibatch = np.ndarray(
            (minibatch_size, 3, insize, insize), np.float32)
         for j in range( minibatch_size):
            idx = i * batchsize + j
            x1, y1, x2, y2 = poslist[idx]
            minibatch[j] = loader.image2array(
               img.crop((x1, y1, x2, y2)).resize((insize, insize))) - self.mean
            # print(x1, y1, x2, y2)
         if self.gpu >= 0:
            minibatch = cuda.to_gpu(minibatch)
         conf = self.model.calc_confidence(minibatch).data
         if self.gpu >= 0:
            conf = cuda.to_cpu(conf)
         if confs is None:
            _, n = conf.shape
            confs = np.ndarray((len(poslist), n), np.float32)
         st = i * batchsize
         #for j in range(st, st + minibatch_size):
         #   confs[j] = conf
         confs[st:st+minibatch_size] = conf
         # print(conf)
      return confs
 
   def sliding_window(self, imgpath, sizes, output, confidence=0.5):
      img = Image.open(imgpath)
      W, H = img.size
      w, h = img.size
      if w > 512:
         h = int(512 * h / w)
         w = 512
         img = img.resize((w, h))
      idx = 0
      sum = 0
      for size, stride in sizes:
         sum += ((w - size + stride - 1) // stride) * ((h - size + stride - 1) // stride)
      print(sum, 'patterns cropped.')
      insize = self.model.insize
      pos = np.ndarray(
         (sum, 4), dtype=np.int32)
      for tuple in sizes:
         size, stride = tuple
         for x in six.moves.range(0, w - size, stride):
            for y in six.moves.range(0, h - size, stride):
               x1 = x
               y1 = y
               x2 = x + size
               y2 = y + size
               pos[idx] = np.array([x1, y1, x2, y2])
               idx += 1
      confs = self.calc_conf_of_subimages(img, pos)
      labels = self.calc_max_label(confs)
      labels, pos = self.filter_images_with_conf(labels, pos, label=0, confidence=confidence)
      labels, pos = self.filter_with_IoU(labels, pos)
      
      pos = self.resize_poslist(pos, W/w)
      self.draw_face_rects(imgpath, output, labels, pos)
      
   def resize_poslist(self, poslist, scale):
      for idx, pos in enumerate(poslist):
         poslist[idx] = pos * scale
      return poslist
         
   def draw_face_rects(self, imgpath, outpath, labels, poslist, color=(128, 255, 64)):
      img = cv2.imread(imgpath)
      for idx, pos in enumerate(poslist):
         _, conf = labels[idx]
         x1, y1, x2, y2 = pos
         cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
         font = cv2.FONT_HERSHEY_DUPLEX
         cv2.putText(img, str(conf*100)[:5]+'%', (x2-120, y1+25), font, 1, (0, 55, 94))
      cv2.imwrite(outpath, img)

   def filter_images_with_conf(self, labels, pos, label=-1, confidence=0.5):
      xs = []
      for idx, tuple in enumerate(labels):
         l, conf = tuple
         if label >= 0 and not l == label:
            continue
         if conf < confidence:
            continue
         xs.append(idx)
      
      f_label = []
      f_pos = np.ndarray(
         (len(xs), 4), dtype=np.int32)
      i = 0
      for x in xs:
         f_label.append(labels[x])
         f_pos[i] = pos[x]
         i += 1
      return f_label, f_pos
         
   def random_crop(self, img, minsize):
      w, h = img.size
      x1 = np.random.randint(0, w-minsize)
      y1 = np.random.randint(0, h-minsize)
      size = np.random.randint(1, min(w-x1, h-y1))
      return img.crop((x1, y1, x1+size-1, y1+size-1))
