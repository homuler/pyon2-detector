from parse import *

def parse_training_log(logpath):
   def parse_logdata(logdata):
      train_data = []
      valid_data = []
      epoch_train_data = []
      epoch_valid_data = []
      for line in logdata:
         if line.startswith('training:'):
            iter = search('iteration={:d}', line)
            loss = search('loss={:f}', line)
            acc  = search('rate={:f}', line)
            epoch_train_data.append({'iteration': iter[0], 'loss': loss[0], 'accuracy':acc[0]})
         elif line.startswith('validation:'):
            iter = search('iteration={:d}', line)
            loss = search('loss={:f}', line)
            acc  = search('rate={:f}', line)
            epoch_valid_data.append({'iteration': iter[0], 'loss': loss[0], 'accuracy':acc[0]})
         elif line.endswith('starts.'):
            train_data.append(epoch_train_data)
            valid_data.append(epoch_valid_data)
            epoch_train_data = []
            epoch_valid_data = []
         else:
            continue
      if len(epoch_train_data) > 0:
         train_data.append(epoch_train_data)
      if len(epoch_valid_data) > 0:
         valid_data.append(epoch_valid_data)
      return train_data, valid_data

   logdata = open(logpath).read().splitlines()
   return parse_logdata(logdata)
