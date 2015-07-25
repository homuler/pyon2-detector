# -*- coding: utf-8 -*-
import sys
import os
import cv2
import numpy as np

argvs = sys.argv
videoPath = argvs[1]
outputDirPath = '../../resources/pictures/samples'
rawDirPath = '../../resources/pictures/raws'
sampleSize = (64, 64) 

cap = cv2.VideoCapture(videoPath)
cascade = cv2.CascadeClassifier('../../resources/cascades/animeface.xml')

def main():
   counter = 0
   frameCount = 0
   root, _ = os.path.splitext(videoPath)
   _, videoName = os.path.split(root)
   print videoName
   while(cap.isOpened()):
      ret, frame = cap.read()
      converted = convertImage(frame)
      faces = cascade.detectMultiScale(converted, scaleFactor=1.1, minNeighbors=4, minSize=(70, 70))
      print faces
      frameCount = frameCount + 1
      if frameCount % 10 != 0:
         continue

      for (x, y, w, h) in faces:
         print x, y, w, h, (x+w), (h+y)
         roi = converted[y:y+h, x:x+w]
         roi2 = frame[y:y+h, x:x+w]
         print str(roi.size), str(converted.size)
         sample = cv2.resize(roi2, sampleSize)
         fileName = videoName + '-' + str(counter) + '.png'
         counter = counter + 1
         cv2.imwrite(os.path.join(outputDirPath, fileName), sample)
         print outputDirPath, '/', fileName, 'saving'
         # cv2.rectangle(converted, (x, y), (x+w, y+h), (255, 255, 255), 3)

      if len(faces) > 0:
         rawFilePath = os.path.join(rawDirPath, 'human', videoName + '-frame' + str(frameCount) + '.png')
      else:
         rawFilePath = os.path.join(rawDirPath, 'nohuman', videoName + '-frame' + str(frameCount) + '.png')
         
      cv2.imwrite(rawFilePath, frame)
      # cv2.imshow('frame', converted)

      if cv2.waitKey(1) > 0:
         break

   cap.release()
   cv2.destroyAllWindows()

def convertImage(frame):
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
   # hist = cv2.equalizeHist(gray, gray)
   return gray

main()
