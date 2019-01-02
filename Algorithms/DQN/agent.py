import os
import sys
import time
import numpy as np
import tensorflow as tf
from Algorithms.agent_base import BaseAgent
from Algorithms.DQN.model import DQNNetwork
from Algorithms.DQN.replay_memory import ReplayMemory

from environment import Atari

option = dict()

option['exploration_steps']  = 50000
option['update_target_rate'] = 10000
option['history_size']       = 4
option['memory_size']        = 400000
option['batch_size']         = 32
option['discount_factor']    = 0.99
option['epsilon_start']      = 1.0
option['epsilon_end']        = 0.1
option['epsilon_steps']      = 1000000
option['double_q']           = False
option['dueling_q']          = False

class Agent(BaseAgent):
  ''' Deep Q-Network (DQN) Agent '''
  def __init__(self, env_name, dirs):  
    
    self.env                = Atari(env_name)

    # 중간 잘나오는지 확인을 위한 validation environment
    self.valid_env          = Atari(env_name)
    self.valid_env.monitor(video_dir = dirs['video'], video_callable=lambda x: True)

    # Model
    self.network            = DQNNetwork(
                                input_shape      = self.env.state_size,
                                output_size      = self.env.action_size,
                                double_q         = option['double_q'],
                                dueling_q        = option['dueling_q'], 
                                discount_factor  = option['discount_factor'])
    self.model_layers       = self.network.model_layers

    # Replay Memory
    self.memory             = ReplayMemory(option['memory_size'])

    # Greedy Parameter
    self.epsilon            = option['epsilon_start']
    self.eps_end            = option['epsilon_end']
    self.eps_decay          = (option['epsilon_start'] - option['epsilon_end']) \
                               / option['epsilon_steps']
   
    # Train Parameter
    self.batch_size         = option['batch_size']
    self.exploration_steps  = option['exploration_steps']
    self.update_target_rate = option['update_target_rate']

    super().__init__(dirs         = dirs, 
                     option       = option, 
                     env          = self.env, 
                     name         = 'DQN', 
                     model_layers = self.model_layers)

  def get_action(self, state):    
    ''' 일정 확률(epsilon)에 따라 임의의 행동 또는 모델에 따른 행동을 반환

    Parameters
    ----------
    state: np.array uint8 [1, height, width, history]
      행동을 할 상태. 

    Returns
    -------
    action: int 
      상태에 따른 행동을 결정한다.[0, self.action_size) 범위의 숫자가 나옴

    '''
    
    if np.random.rand() <= self.epsilon:
      action = self.env.random_action()
    else:
      action = self.sess.run(self.network.argmax_q, 
                            feed_dict={self.network.inputs:state})[0]

    return action
  
  def train_model(self):
    ''' learner DQN 모델을 학습. memory로부터 state, actions, rewards,
        next_states, dones를 임의로 가져와서 build_loss에서 만든대로 학습.
    '''
    if self.epsilon > self.eps_end:
      print('decay')
      self.epsilon -= self.eps_decay                                            
    
    states, actions, rewards, next_states, dones = \
        self.memory.sample(self.batch_size)

    return self.network.train(sess        = self.sess, 
                              states      = states,
                              actions     = actions,
                              rewards     = rewards,
                              next_states = next_states,
                              dones       = dones)

  def train(self):

    # 나중에 validation이 몇 번 했는지 위한 counter
    valid_step = 0

    # 처음 시작할 때 learner DQN과 target DQN의 가중치를 동일하게 한다.
    self.network.update_target_dqn(self.sess)

    # 학습 시작
    # ==============================================================
    total_step = 0
    for episode in range(50000):

      state = self.env.reset()

      avg_q, avg_loss, done = 0, 0, False
      while not done:

        # 1. 현재 상태에 따라 행동 선택
        action = self.get_action([state])

        # 2. 관찰 : 다음 상태, 보상, 끝, 죽음 확인
        next_state, reward, done, dead = self.env.step(action)

        avg_q += self.sess.run(self.network.max_q, 
                              feed_dict={self.network.inputs:[state]})[0]

        # 3. 메모리에 저장
        self.memory.save(state, action, reward, next_state, dead)

        # 4. 학습 
        # - 일정기간동안 탐험을 하여 메모리를 쌓은 후에 실시
        if total_step >= self.exploration_steps:
          avg_loss += self.train_model()

          # 5. 일정 주기마다 타겟 네트워크에 가중치 복사
          if total_step % self.update_target_rate == self.update_target_rate - 1:
            self.network.update_target_dqn(self.sess)
        
        # Go Next
        state = next_state
        total_step += 1 

      if total_step < self.exploration_steps:
        sys.stdout.write(f"Explorations...{total_step}\r")
      
      else:
        steps = self.env.steps
        score = self.env.score

        lr = self.sess.run(self.network.lr)

        self.summary.write(sess = self.sess, step = episode,
                          summary_dict = {
                            "Loss" : avg_loss / steps,
                            "Max_q" : avg_q / steps,
                            "Learning_rate": lr,
                            "Steps": steps,
                            "Score": score,
                            "Memory_size": len(self.memory),
                            "Epsilon": self.epsilon})

        self.logger.info(f"Episode {episode:5} | "
                        f"Steps {steps:5} | "
                        f"Total Step {total_step:8} | "
                        f"Score {score:3} | "
                        f"Avg_q {avg_q/steps:.5f} | "
                        f"Avg_loss {avg_loss/steps:.3f} | "
                        f"Memory size {len(self.memory):6} | "
                        f"Epsilon: {self.epsilon:.3f} | ")
        
        if episode % 20 == 0:
          state = self.valid_env.reset()
          valid_step += 1

          done = False
          while not done:
            if np.random.rand() <= 0.05:
              action = self.valid_env.random_action()
            else:
              action = self.sess.run(self.network.argmax_q, 
                                    feed_dict={self.network.inputs:[state]})[0]
            
            next_state, reward, done, _ = self.valid_env.step(action)

            state = next_state

          self.logger.info(f"Validation Step : {valid_step - 1}, Score : {self.valid_env.score}")
