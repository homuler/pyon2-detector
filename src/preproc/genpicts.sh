#!/bin/bash

while read line;
do
   python2 genpicts.py $line;
done;

