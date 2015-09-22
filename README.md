# Pyon2-Detector
Scripts to train "Anime face detector" model using deep learning, and detect anime faces with the trained model.  
For more details, please refer to [this page](http://qiita.com/homulerdora/items/9a9af1481bf63470731a).

## Prepare Dataset
place dataset like this.
```
  dataset
   /train
     /0      -- 0 labeled dataset
     /1      -- 1 labeled dataset
     ...
   /valid
     /0      -- 0 labeled dataset
     /1      -- 1 labeled dataset
     ...
```

## Usage
```shell
# resize images
./pyon2-detector.py gendata --size=[output image size] --rawdata=[dataset directory] --output=[resized dataset directory]

# generate mean file
./pyon2-detector.py genmean --train=[training dataset directory] --output=[mean file path]

# generate train data list and validation data list
./pyon2-detector.py genlist --dataset=[training data directory] --output=[training data list]
./pyon2-detector.py genlist --dataset=[validation data directory] --output=[validation data list]

# train model
./pyon2-detector.py learn --train=[training data list] --valid=[validation data list] --gpu=[gpu flag] --mean=[mean file path] --model=[model name]

# detect faces
./pyon2-detector.py detect --gpu=[gpu flag] --mean=[mean file path] --model=[model dump file path] --input=[input image path] --output=[output image path]
```

## Reqirements
1. chainer
1. OpenCV >= 3.0

