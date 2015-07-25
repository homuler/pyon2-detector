#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import os
import cv2
import random

def main():
   argvs = sys.argv
   imgPath = argvs[1]
   print imgPath
   n = int(argvs[2])
   outputDirPath = '../../resources/pictures/test/valid/1'
   sampleSize = (250, 250) 

   root, _ = os.path.splitext(imgPath)
   _, imgName = os.path.split(root)
   print 'cropping', imgName
   imgCount = 0
   while imgCount < n:
      img = cv2.imread(imgPath)
      randImg = randomCrop(img, sampleSize)
      filePath = os.path.join(outputDirPath, imgName + '-rand-' + str(imgCount) + '.png')
      cv2.imwrite(filePath, randImg)
      imgCount += 1

def randomCrop(img, size):
   maxh, maxw, _ = img.shape
   sampleh, samplew = size
   x = random.uniform(0, maxw-samplew/2)
   y = random.uniform(0, maxh-sampleh/2)
   s = random.uniform(min(samplew, sampleh)/2, min(maxh-y, maxw-x))
   cropped = img[y:y+s, x:x+s]
   return cropped

main()
