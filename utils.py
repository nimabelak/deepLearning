import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

class utils:
    def __init__(self):
      self.x_train = None
      self.y_train = None
      self.x_test = None
      self.y_test = None
    def loadData(self):
      mnist = tf.keras.datasets.mnist
      (self.x_train,self.y_train),(self.x_test,self.y_test) = mnist.load_data()
    def plotSomeData(self):
      idx = np.random.choice(60000 , 9) #returns 9 random numbers between 0 and 60000
      imgs = self.x_train[idx,:,:]  # 1 number in idx , all 28 in axis 0 and all 28 in axis 1.
      _,axis = plt.subplots(3,3 , figsize=(12,12))# _ means we don't care about the first returned value.
      axis = axis.flatten()
      for i,ax in enumerate(axis):
        ax.imshow(imgs[i])
      plt.show()