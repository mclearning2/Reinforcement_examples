import os
import numpy as np
import tensorflow as tf

class ModelBuilder():
  def __init__(self):
    self.layers = []

  def preprocess(self, 
                 normalize=False,
                 data_format='channels_first' if tf.test.is_gpu_available() \
                        else 'channels_last'):

    def wrapper(x):

      self.layers.append(x)

      if normalize:
        x = tf.divide(x, 255., 'normalized_input')
        self.layers.append(x)
    
      if data_format == 'channels_first':
        x = tf.transpose(x, perm=[0, 3, 1, 2], name='channels_first_input')
        
        self.layers.append(x)

      return x

    return wrapper

  def conv2d(self, 
             filters,
             kernel_size=3,
             strides=1,
             activation=None,
             padding='valid',
             data_format='channels_first' if tf.test.is_gpu_available() \
                    else 'channels_last',
             kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
             name = None,
             *args,
             **kargs):
    ''' Convolution for 2D

    Parameters
    ----------
      filters: int
        Feature map 개수
      kernel_size: int or list(shape=[2])
        커널 크기
      strides: int
        Stride 크기
      activation: function
        Activation function (ex. tf.nn.relu, tf.nn.softmax)        
      padding: int or list(shape=[2])
        패딩 크기
      data_format: str
        - 'channels_first' : [batch size, channel size, height, width]
        - 'channels_last' : [batch size, height, width, channel size]
      kernel_initializer: tensor
        Initialize for weight matrix
        ex) tf.contrib.layers.xavier_initializer_conv2d()
      name: str
        Tensor Name
    '''

    def wrapper(x):
      x = tf.layers.conv2d(
              inputs             = x, 
              filters            = filters,
              kernel_size        = kernel_size, 
              strides            = strides, 
              activation         = activation,
              padding            = padding, 
              data_format        = data_format,      
              kernel_initializer = kernel_initializer,
              name               = name,
              *args,
              **kargs)

      self.layers.append(x) 

      return x

    return wrapper
    
  def dense(self, 
            units, 
            activation         = None, 
            kernel_initializer = tf.variance_scaling_initializer(scale=2.0), 
            name               = None,
            *args, 
            **kargs):
    ''' Dense (fully connected layer)

    Parameters
    ----------
      units: int
        The number of output neurons
      activation: function
        Activation function (ex. tf.nn.relu, tf.nn.softmax)        
      kernel_initializer: tensor
        Initialize for weight matrix
        ex) tf.contrib.layers.xavier_initializer()
      name: str
        Tensor Name
    '''

    def wrapper(x):
      x = tf.layers.dense(
        inputs             = x,
        units              = units,
        activation         = activation,
        kernel_initializer = kernel_initializer,
        name               = name,
        *args,
        **kargs)

      self.layers.append(x)
      return x            

    return wrapper

  def flatten(self, x, name="flatten"):
    ''' 3D -> 1D '''
    x = tf.layers.flatten(x, name)

    self.layers.append(x)

    return x

  def loss_func(self, loss_type, pred, reference, name="loss"):
    ''' loss function
    
    Parameters
    ----------
      loss_type: str:
        loss 종류 ["L1", "MSE", "HUBER" ]
      pred: tensor
        prediction(예측 값)
      reference: tensor
        비교하기 위한 reference
    '''
    if loss_type == 'L1':
      loss = tf.losses.absolute_difference(reference, pred)
    elif loss_type == 'MSE':
      loss = tf.losses.mean_squared_error(reference, pred)
    elif loss_type == 'HUBER':
      loss = tf.losses.huber_loss(reference, pred)
    else:
      raise ValueError("Loss is not appropriate, Loss is", loss_type)
    
    self.layers.append(loss)

    return loss

  def optimizer(self, 
                optim_type,
                learning_rate = 0.001, 
                *args, 
                **kargs):
    
    if optim_type == 'Adam':
      optim = tf.train.AdamOptimizer(learning_rate, *args, **kargs)
    elif optim_type == 'RMSProp':
      optim = tf.train.RMSPropOptimizer(learning_rate, *args, **kargs)
  
    self.layers.append(optim)

    return optim
                      
  def grads(self, optimizer, loss, var_list=None):
    grads_and_vars = optimizer.compute_gradients(loss, var_list)
    grads = [grad for grad, var in grads_and_vars 
                      if grad is not None]
    return grads

  def vars(self, optimizer, loss, var_list=None):
    grads_and_vars = optimizer.compute_gradients(loss, var_list)
    vars = [var for grad, var in grads_and_vars 
                      if grad is not None]
    return vars

  def grads_and_vars(self, optimizer, loss, var_list=None):
    g_and_v = optimizer.compute_gradients(loss, var_list)
    g_and_v = [[grad, var] for grad, var in g_and_v 
                      if grad is not None]
    return g_and_v

  def clip_grad(self, 
                grads,
                clip_type,
                clip_param,
                *args,
                **kargs):
    if clip_type == 'GlobalNorm':
      return tf.clip_by_global_norm(grads, clip_param)
    elif clip_type == 'Norm':
      return tf.clip_by_norm(grads, clip_param)
    elif clip_type == 'AverageNorm':
      return tf.clip_by_average_norm(grads, clip_param)
    elif clip_type == 'Value':
      return tf.clip_by_value(grads, clip_param[0], clip_param[1])
    else:
      raise ValueError(f'Clip type is not appropriate. you feed {clip_type}\n'
                       '[GlobalNorm, Norm, AverageNorm, Value]')
      
  def train_op(self, 
               optim_type,
               loss,
               learning_rate=0.001,
               var_list=None,
               *args,
               **kargs):

    optim = self.optimizer(optim_type, learning_rate, *args, **kargs)            

    train_op = optim.minimize(loss, 
                              var_list=var_list,
                              global_step=tf.train.get_or_create_global_step())

    return train_op