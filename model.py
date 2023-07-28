import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import tensorflow.python.keras.layers as ksl


class Model:
  def __init__(self):
    self.net = None
    self.inputShape = [784]
  def buildModel(self):
    self.net = tf.keras.models.Sequential([ksl.Dense(256 , activation = "tanh", #256 is the dimension of output
                                                  input_shape = self.inputShape),
                                        ksl.Dense(128,activation='tanh'),
                                        ksl.Dense(64,activation = 'tanh'),
                                        ksl.Dense(10,activation='softmax')]) #implements 1 fully connected Dense layer.

  def compileModel(self):
    tf.keras.utils.plot_model(self.net,'model.png')
    # self.net.summary()
    loss = tf.keras.losses.CategoricalCrossentropy()
    optim = tf.keras.optimizers.SGD(learning_rate = 0.01)
    # self.net.compile(loss = 'categorical_crossentropy')
    self.net.compile(loss=loss)