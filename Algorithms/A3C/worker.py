import scipy.signal
import numpy as np
import tensorflow as tf

from environment import Atari

global episode
episode = 0

class Worker():
  def __init__(self, env, net, discount_factor, global_net):
    self.env             = env
    self.local_net       = net
    self.discount_factor = discount_factor
    self.global_net      = global_net
    
    self.copy_op = self.get_copy_network_op(global_net.scope_name, net.scope_name)

  def get_copy_network_op(self, src_scope, dest_scope):
    ''' learner DQN의 가중치들을 target DQN 가중치로 복사하기 위한 텐서 반환 '''
    
    op_holder = []

    src_vars = tf.trainable_variables(src_scope)
    dest_vars = tf.trainable_variables(dest_scope)
    
    for s, d in zip(src_vars, dest_vars):
      op_holder.append(d.assign(s.value()))

    return op_holder

  def update_local_network(self):
    self.sess.run(self.copy_op)

  def discount(self, x, discount_factor):
    # for i in reversed(range(len(x) - 1)):
    #   x[i] += x[i+1] * discount_factor
    return scipy.signal.lfilter([1], [1, -discount_factor], x[::-1], axis=0)[::-1]
    
  def get_action(self, state):
    policy = self.local_net.eval_policy(self.sess, state)
    policy = np.squeeze(policy)
    action = np.random.choice(self.env.action_size, 1, p=policy)[0]

    self.avg_max_prob += np.max(policy)

    return action    

  def save_memory(self, update_cnt, state, action, reward):
    state = np.expand_dims(np.array(state), axis=0)

    if update_cnt == 0:
      self.states  = state
      self.actions = np.array([action])
      self.rewards = np.array([reward])
    else:
      self.states  = np.concatenate((self.states, state), axis=0)
      self.actions = np.hstack((self.actions, action))
      self.rewards = np.hstack((self.rewards, reward))

  def train_model(self, done, last_state):
        
    if not done:
      bootstrap_value = self.local_net.eval_value(self.sess, [last_state])[0]
    else:
      bootstrap_value = 0.
    
    rewards_plus = np.append(self.rewards, bootstrap_value)
    discount_reward = self.discount(rewards_plus, self.discount_factor)[:-1]
    
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    values     = self.local_net.eval_value(self.sess, self.states)
    value_plus = np.append(values, bootstrap_value)
    advantage = self.rewards + self.discount_factor * value_plus[1:] - value_plus[:-1]
    advantage = self.discount(advantage, self.discount_factor)
  
    loss, _ = self.sess.run([self.local_net.loss,
                             self.local_net.train_op], 

                             feed_dict = {self.local_net.inputs: self.states,
                                          self.local_net.actions: self.actions,
                                          self.local_net.target_value: discount_reward,
                                          self.local_net.advantages: advantage})
  
    return loss

  def run(self, sess, summary, logger, coord, update_rate):
    global episode

    self.sess  = sess
    self.update_local_network()

    with self.sess.graph.as_default():
      while not coord.should_stop():
        try:
          self.avg_max_prob = 0
          update_cnt = 0
          done = False

          state = self.env.reset()
          while not done:
            
            action = self.get_action([state])

            next_state, reward, done, dead = self.env.step(action)

            self.save_memory(update_cnt, state, action, reward)
            
            update_cnt += 1
            if update_cnt >= update_rate or done:
              loss = self.train_model(done, last_state = state)
              self.update_local_network()
              update_cnt = 0
              
            state = next_state

          episode += 1

          summary.write(sess = self.sess, step = episode,
                        summary_dict = {
                        "Score" : self.env.score,
                        "Steps": self.env.steps,
                        "Max_prob" : self.avg_max_prob / self.env.steps,
                        "Learning_rate" : self.local_net.lr,
                        "Loss" : loss,
                        })
          logger.info(f"Episode {episode - 1} | " # Video랑 맞추기위해 -1
                      f"Steps {self.env.steps} | "
                      f"Score {self.env.score} | "
                      f"Average max prob {self.avg_max_prob/self.env.steps:.3f}")

        except KeyboardInterrupt:
          coord.request_stop()



