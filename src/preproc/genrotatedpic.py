# -*- coding: utf-8 -*-
import sys
import os
import cv2
import random

def main():
   argvs = sys.argv
   imgPath = argvs[1]
   outDirName = argvs[2]
   print imgPath
   outputDirPath = os.path.join('../../resources/pictures/test/size64/train', outDirName)

   root, _ = os.path.splitext(imgPath)
   _, imgName = os.path.split(root)
   print imgName
   img = cv2.imread(imgPath)
   w, h, _ = img.shape

   filePath0 = os.path.join(outputDirPath, imgName + '-r0.png')
   print filePath0
   cv2.imwrite(filePath0, img)
   
   mat1 = cv2.getRotationMatrix2D((w/2, h/2), 90, 1)
   r1 = cv2.warpAffine(img, mat1, (w, h))
   filePath1 = os.path.join(outputDirPath, imgName + '-r90.png')
   cv2.imwrite(filePath1, r1)

   mat2 = cv2.getRotationMatrix2D((w/2, h/2), 180, 1)
   r2 = cv2.warpAffine(img, mat2, (w, h))
   filePath2 = os.path.join(outputDirPath, imgName + '-r180.png')
   cv2.imwrite(filePath2, r2)

   mat3 = cv2.getRotationMatrix2D((w/2, h/2), 270, 1)
   r3 = cv2.warpAffine(img, mat3, (w, h))
   filePath3 = os.path.join(outputDirPath, imgName + '-r270.png')
   cv2.imwrite(filePath3, r3)

main()
