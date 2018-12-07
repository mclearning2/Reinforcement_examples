# Reference : https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb

import os
import time
import threading
import multiprocessing
import numpy as np
import tensorflow as tf

from Algorithms.agent_base import BaseAgent
from Algorithms.A3C.model import A3CNetwork
from Algorithms.A3C.worker import Worker

from environment import Atari

option = dict()

option['n_thread']        = multiprocessing.cpu_count()
option['update_rate']     = 20
option['discount_factor'] = 0.99

class Agent(BaseAgent):
  ''' Asynchronouse Advantage Actor-Critic (A3C) Agent '''
  def __init__(self, env_name, dirs):

    self.env        = Atari(env_name)
    
    self.global_net = A3CNetwork(input_shape = self.env.state_size, 
                                 output_size = self.env.action_size, 
                                 scope_name  = 'global_network')

    self.workers = list()
    for t in range(option['n_thread']):
      
      net    = A3CNetwork(input_shape  = self.env.state_size, 
                          output_size  = self.env.action_size, 
                          scope_name   = 'local_network_' + str(t),
                          global_scope = self.global_net.scope_name)
      
      env    = Atari(env_name)
      if t == 0:
        env.monitor(dirs['video'], video_callable=lambda x : x > (10000 / option['n_thread']))

      worker = Worker(env             = env,
                      net             = net,
                      discount_factor = option['discount_factor'],
                      global_net      = self.global_net)

      self.workers.append(worker)

    super().__init__(dirs         = dirs,
                     option       = option, 
                     env          = self.env,
                     name         = 'A3CAgent', 
                     model_layers = self.global_net.model_layers)

  def train(self):
    
    try:
      coord = tf.train.Coordinator()

      threads = list()
      for t, worker in enumerate(self.workers):

        thread = threading.Thread(target  = 
                    lambda: worker.run(sess        = self.sess,
                                       summary     = self.summary,
                                       logger      = self.logger,
                                       coord       = coord, 
                                       update_rate = option['update_rate'])
                                 )
        thread.daemon = True
        thread.start()
        
        self.logger.info(f"worker_{t} starts.")

        threads.append(thread)
        time.sleep(1)

      coord.join(threads)

    except KeyboardInterrupt:
      coord.join(threads)