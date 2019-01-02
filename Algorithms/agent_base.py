import os
import logging
import logging.handlers
import tensorflow as tf

from abc import ABCMeta, abstractmethod
from Common.summary import Summary

class BaseAgent(metaclass=ABCMeta):
  ''' Agent들의 기반. learn(학습)과 play(실제 게임하기)로 구성되어있다.

  Parameters
  ----------
  trian : function
    학습을 구현

  test : function
    학습된 agent로 실제 게임 테스트

  env : gym object
    Atari game의 make된 환경. 정보를 출력하기 위해 사용. 실제 학습에 사용하지 않아도 됨.

  name : str
    agent의 이름

  model_layers : list
    Tensorflow model build하여 생성한 출력물(layer)들을 저장한 리스트 

  
  Result
  ------
  self.sess : tf.Session()

  self.env : gym object

  self.name : str

  self.model_layers : list

  self.dirs : dict
  
  self.option : dict

  self.logger : logging

  self.saver : tf.train.Saver()
  
  self.summary : tf.summary
  
  '''

  def __init__(self, dirs, option, env, name, model_layers):

    self.sess         = tf.Session()
    self.env          = env
    self.name         = name
    self.model_layers = model_layers
    self.dirs         = dirs
    self.option       = option
  
  def _generate_logger(self, filename, 
                        maxBytes=1024 * 1024 * 10, #10MB
                        backupCount=10):
    logger = logging.getLogger()

    file_handler = logging.handlers.RotatingFileHandler(filename, maxBytes, backupCount)
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.setLevel(logging.INFO)

    return logger

  def _set_summary(self, summary_dir):
    summary = Summary(summary_dir, self.sess, self.env.name)
    self.logger.info('Summary initialized')

    return summary

  def _set_saver(self, saver_dir):
    saver = tf.train.Saver()
    self.logger.info('Saver initialized')
    recent_ckpt_job_path = tf.train.latest_checkpoint(saver_dir)
          
    if recent_ckpt_job_path is None:
      self.sess.run(tf.global_variables_initializer())
      self.logger.info("Initializing variables...")
    else:
      saver.restore(self.sess, recent_ckpt_job_path)
      self.logger.info(f"Restoring...{recent_ckpt_job_path}")

    return saver

  def _write_info(self):
    self.logger.info(f"\n")
    self.logger.info(f"Saver will be saved in {self.dirs['checkpoint']}")
    self.logger.info(f"Summary will be saved in {self.dirs['summary']}")
    self.logger.info(f"Video will be saved in {self.dirs['video']}")
    self.logger.info(f"Logging will be saved in {self.dirs['logs']}")

    self.logger.info("\n")
    self.logger.info("Environment >")
    self.logger.info("=" * 100)
    self.logger.info(f"Environment name : {self.env.name}")
    self.logger.info(f"Action types : {self.env.valid_actions}")
    self.logger.info(f"State shape : {self.env.state_size}")
    self.logger.info("=" * 100)

    self.logger.info("\n")
    self.logger.info(f"Agent is {self.name}")

    self.logger.info("\n")
    self.logger.info("Option")
    self.logger.info("=" * 100)
    for key, value in self.option.items():
      self.logger.info(f"{key} : {value}")      
    self.logger.info("=" * 100)
        
    self.logger.info("\n")
    self.logger.info("Model Layers")
    self.logger.info("=" * 100)
    for layer in self.model_layers:
      self.logger.info(f"{layer}")
    self.logger.info("=" * 100)

  def _before_train(self):
    self.logger  = self._set_logger(os.path.join(self.dirs['logs'], 'train.log'))
    self.summary = self._set_summary(os.path.join(self.dirs['summary']))
    self.saver   = self._set_saver(self.dirs['checkpoint']) 

    self._write_info()

  def _after_train(self):
    self.logger.info("Saving Model...")
    checkpoint_path = os.path.join(self.dirs['checkpoint'], "model.ckpt")
    self.saver.save(self.sess, checkpoint_path, 
                    global_step=tf.train.get_global_step())
    
    self.logger.info("Closing session...")
    self.sess.close()

    self.logger.info("Reseting default graph...")
    tf.reset_default_graph()
    self.logger.info("Successfuly Done")

  def learn(self):
    self._before_train()
    try:
      self.logger.info("\n")
      self.logger.info("Training starts")
      self.train()
    except KeyboardInterrupt:
      self.logger.info("Training is terminated")

    self._after_train()

  @abstractmethod
  def train(self):
    pass
  