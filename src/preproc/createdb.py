#!/usr/bin/python2
# -*- coding: utf-8 -*-
import sys
import shutil
import subprocess
import os
import random
from caffe.proto import caffe_pb2
import leveldb
import numpy as np
from PIL import Image

resourcePath = '../../resources'
sampleSize = 64 

def createLevelDB(name):
   dbPath = os.path.join(resourcePath, 'db', name)
   try:
      shutil.rmtree(dbPath)
   except OSError:
      pass
   return leveldb.LevelDB(
      dbPath, create_if_missing=True, error_if_exists=True, paranoid_checks=True
   )

def genSampleInfoList(dirPath):
   xs = []
   for p, _, filenames in os.walk(dirPath):
      try:
         _, l = os.path.split(p)
         label = int(l)
      except Exception:
         continue
      
      for filename in filenames:
         if filename.endswith(('.png', '.jpg')):
            xs.append((os.path.join(p, filename), label))

   return xs

def makeDatum(image, label):
   return caffe_pb2.Datum(
      channels=3,
      width=sampleSize,
      height=sampleSize,
      label=label,
      data=np.rollaxis(np.asarray(image), 2).tostring()
   )

def main():
   inputDirPath = os.path.join(resourcePath, 'pictures/test')
   outputDirPath = os.path.join(resourcePath, 'pictures/test/random')
#   n = randomPick(12, os.path.join(inputDirPath, 'correct'), os.path.join(outputDirPath, '0'))
#   m = randomPick(2, os.path.join(inputDirPath, 'incorrect'), os.path.join(outputDirPath, '1'))

#   print 'correct data = ', n
#   print 'incorrect data = ', m
   
   trainDB = createLevelDB('animeface_train_leveldb')
   testDB = createLevelDB('animeface_test_leveldb')
   
   res = genSampleInfoList(outputDirPath)
   random.shuffle(res)

   for seq, (filepath, label) in enumerate(res):
      print seq, label, filepath

      try:
         image = Image.open(filepath)
      except:
         continue

      datum = makeDatum(image.convert('RGB'), label)
      db = testDB if seq % 10 == 0 else trainDB
      db.Put('%08d' % seq, datum.SerializeToString())

def randomPick(n, fromDir, toDir):
   count = 0
   files = os.listdir(fromDir)
   print 'path = ', fromDir
   for file in files:
      r = random.uniform(0, n)
      if int(r) == 0:
        shutil.copyfile(os.path.join(fromDir, file), os.path.join(toDir, file))
        count += 1
   
   return count

main()
