#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import cv2
import numpy as np

parser = argparse.ArgumentParser(
    description='Detecting faces and save pictures.')
parser.add_argument('option', help='crop face pictures.')
parser.add_argument('--type', '-t', default='video',
                    help='Input file type \
                    (video, picture)')
parser.add_argument('--filepath', '-f',
                    help='input file')
parser.add_argument('--interval', '-i', default=10, type=int,
                    help='skip frame interval')
parser.add_argument('--size', '-s', default=64, type=int,
                    help='minimul face size')
parser.add_argument('--out', '-o', default='.',
                    help='Path to save pictures')

cascade = cv2.CascadeClassifier('../../../resources/cascades/animeface.xml')

def main():
   args = parser.parse_args()
   if args.type == 'video':
      detect_faces_from_videos(args.filepath, args.out, args.size, args.interval)
   elif args.option == 'detect':
      detect_face_rects(args.filepath, args.out)
   elif os.path.isdir(args.filepath):
      detect_faces_from_dir(args.filepath, args.out, args.size)
   else:
      print('hoge')
      detect_faces_from_pict(args.filepath, args.out, args.size)
  
def save_pict(frame, file_name, output_dir):
   if not os.path.exists(output_dir):
      os.makedirs(output_dir)
   cv2.imwrite(os.path.join(output_dir, file_name), frame)
   print('saving ', file_name, 'to', output_dir)

def detect_face_rects(imgpath, outpath):
   frame = cv2.imread(imgpath)
   faces = cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4, minSize=(64, 64))

   for (x, y, w, h) in faces:
      roi = frame[y:y+h, x:x+w]
      cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 255, 64), thickness=2)

   cv2.imwrite(outpath, frame)

def detect_faces_from_frame(frame, frame_name, output_dir, min_size):
   counter = 0
   faces = cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=0, minSize=(min_size, min_size))

   for (x, y, w, h) in faces:
      roi = frame[y:y+h, x:x+w]
      file_name = frame_name + '-' + str(counter) + '.png'
      counter = counter + 1
      save_pict(roi, file_name, output_dir)

   return len(faces) > 0

def detect_faces_from_pict(pict_path, output_dir, min_size):
   print('reading ', pict_path)
   frame = cv2.imread(pict_path)

   root, _ = os.path.splitext(pict_path)
   _, pict_name = os.path.split(root)
   detected = detect_faces_from_frame(frame, pict_name, output_dir, min_size)
   if detected:
      save_pict(frame, pict_name + '.png', os.path.join(output_dir, 'human'))
   else:
      save_pict(frame, pict_name + '.png', os.path.join(output_dir, 'nohuman'))

def detect_faces_from_dir(dir_path, output_dir, min_size):
   files = os.listdir(dir_path)

   for file in files:
      fullpath = os.path.join(dir_path, file)
      if os.path.isdir(fullpath):
         continue
      detected = detect_faces_from_pict(fullpath, output_dir, min_size)

def detect_faces_from_videos(video_path, output_dir, min_size, interval):   
   cap = cv2.VideoCapture(video_path)
   frame_count = 0
   root, _ = os.path.splitext(video_path)
   _, video_name = os.path.split(root)
   while(cap.isOpened()):
      ret, frame = cap.read()
      
      if not ret:
         break
      frame_count = frame_count + 1
      if frame_count % interval != 0:
         continue

      frame_name = video_name + '-' + str(frame_count) 
      detected = detect_faces_from_frame(frame, frame_name, output_dir, min_size)
      if detected:
         save_pict(frame, frame_name + '.png', os.path.join(output_dir, 'human'))
      else:
         save_pict(frame, frame_name + '.png', os.path.join(output_dir, 'nohuman'))
      cv2.imshow('window', frame)
      if cv2.waitKey(1) > 0:
         break

   cap.release()

main()
