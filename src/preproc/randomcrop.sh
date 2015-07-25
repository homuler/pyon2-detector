#!/bin/bash

DIR=$1
COUNT=$2

for img in ${DIR}/*.png;
do
   ./python/randomcrop.py ${img} $COUNT;
done;
