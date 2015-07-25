#!/bin/bash

DIR=$1
TYPE=$2

for img in ${DIR}/*.png;
do
   python2 genrotatedpic.py ${img} ${TYPE};
done;
