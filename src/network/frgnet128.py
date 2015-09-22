import chainer
import chainer.functions as F

class FrgNet128(chainer.FunctionSet):
   """ Single-GPU Network
   """

   insize = 128 

   def __init__(self):
      super(FrgNet128, self).__init__(
         conv1 = F.Convolution2D(3, 128, 7, pad=3),
         bn1   = F.BatchNormalization(128),
         conv2 = F.Convolution2D(128, 256, 5, pad=2),
         bn2   = F.BatchNormalization(256),
         conv3 = F.Convolution2D(256, 256, 5, pad=2),
         conv4 = F.Convolution2D(256, 512, 3, pad=1),
         conv5 = F.Convolution2D(512, 512, 3, pad=1),
         fc6   = F.Linear(4608, 4096),
         fc7   = F.Linear(4096, 4096),
         fc8   = F.Linear(4096, 2),
      )

   def forward_but_one(self, x_data, train=True):
      x = chainer.Variable(x_data, volatile=not train)
		
      h = F.max_pooling_2d(F.relu(self.bn1(self.conv1(x))), 5, stride=3)
      h = F.max_pooling_2d(F.relu(self.bn2(self.conv2(h))), 5, stride=3)
      h = F.max_pooling_2d(F.relu(self.conv3(h)), 3, stride=2)
      h = F.max_pooling_2d(F.relu(self.conv4(h)), 3, stride=2)
      h = F.relu(self.conv5(h))
      h = F.dropout(F.relu(self.fc6(h)), train=train)
      h = F.dropout(F.relu(self.fc7(h)), train=train)
      return self.fc8(h)
     
   def calc_confidence(self, x_data, y_data):
      h = self.forward_but_one(x_data, train=False)
      return F.softmax(h)

   def forward(self, x_data, y_data, train=True):
      """ You must subtract mean value from the dataset before. """
      y = chainer.Variable(y_data, volatile=not train)
      h = self.forward_but_one(x_data, train=train)
      return F.softmax_cross_entropy(h, y), F.accuracy(h, y) 
