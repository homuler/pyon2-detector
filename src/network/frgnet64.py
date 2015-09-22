import chainer
import chainer.functions as F

class FrgNet64(chainer.FunctionSet):
   """ Single-GPU Cifar10 Network
   """

   insize = 64

   def __init__(self):
      super(FrgNet64, self).__init__(
         conv1 = F.Convolution2D(3, 96, 5, pad=2),
         bn1   = F.BatchNormalization(96),
         conv2 = F.Convolution2D(96, 128, 5, pad=2),
         bn2   = F.BatchNormalization(128),
         conv3 = F.Convolution2D(128, 256, 3, pad=1),
         conv4 = F.Convolution2D(256, 384, 3, pad=1),
         fc5 = F.Linear(18816, 2048),
         fc6   = F.Linear(2048, 2),
      )

   def forward_but_one(self, x_data, train=True):
      x = chainer.Variable(x_data, volatile=not train)
		
      h = F.max_pooling_2d(F.relu(self.bn1(self.conv1(x))), 5, stride=2)
      h = F.max_pooling_2d(F.relu(self.bn2(self.conv2(h))), 5, stride=2)
      h = F.max_pooling_2d(F.relu(self.conv3(h)), 3, stride=2)
      h = F.leaky_relu(self.conv4(h), slope=0.2)
      h = F.dropout(F.leaky_relu(self.fc5(h), slope=0.2), train=train)
      return self.fc6(h)

   def calc_confidence(self, x_data):
      h = self.forward_but_one(x_data, train=False)
      return F.softmax(h)

   def forward(self, x_data, y_data, train=True):
      """ You must subtract mean value from the dataset before. """
      y = chainer.Variable(y_data, volatile=not train)
      h = self.forward_but_one(x_data, train=train)
      return F.softmax_cross_entropy(h, y), F.accuracy(h, y) 
