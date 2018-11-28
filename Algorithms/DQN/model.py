import logging
import numpy as np
import tensorflow as tf

from Common.modelbuilder import ModelBuilder

class DQNNetwork:
  def __init__(self, input_shape, 
                     output_size,
                     double_q         = False, 
                     dueling_q        = False,
                     discount_factor  = 0.99):
    ''' DQN을 위한 Network. learner network와 target network 두 가지를 가지고 있다.

    Parameters
    ----------
    input_shape: list or tuple
      상태 shape [height, width, channels]
    
    output_size: int
      행동 가짓수

    double_q: bool
      double q-learning 알고리즘 적용

    dueling_q: bool
      dueling q-learning 알고리즘 적용
    
    discount_factor: float
      감가율

    '''
    self.name             = 'DQN_Network'
    self.learn_dqn_scope  = self.name + '/learner'
    self.target_dqn_scope = self.name + '/target'

    self.input_shape = input_shape
    self.output_size = output_size

    self.build_model(input_shape      = input_shape, 
                     output_size      = output_size, 
                     double_q         = double_q,
                     dueling_q        = dueling_q,
                     discount_factor  = discount_factor)

    self.copy_op = self.get_copy_network_op()

  def train(self, sess, states, actions, rewards, next_states, dones):

    loss, _ = sess.run([self.loss, self.train_op],
              feed_dict = {self.inputs: states,
                           self.actions: actions,
                           self.rewards: rewards,
                           self.t_inputs:next_states,
                           self.dones: dones})

    return loss

  def get_copy_network_op(self):
    ''' learner DQN의 가중치들을 target DQN 가중치로 복사하기 위한 텐서 반환 '''
    
    op_holder = []

    src_vars = tf.trainable_variables(self.learn_dqn_scope)
    dest_vars = tf.trainable_variables(self.target_dqn_scope)
    
    for s, d in zip(src_vars, dest_vars):
      op_holder.append(d.assign(s.value()))

    return op_holder

  def update_target_dqn(self, sess):
    ''' learner의 Network 가중치를 target Network 가중치에 복사'''
    sess.run(self.copy_op)

    # 복사가 제대로 되었는지 검증
    # =============================================================================
    random_input = np.ones([1] + self.inputs.get_shape().as_list()[1:])

    q, t_q = sess.run([self.q, self.t_q], feed_dict={self.inputs: random_input, 
                                            self.t_inputs: random_input})

    if np.all(np.equal(q, t_q)):
      logging.info("Successfully target network updated")
    else:
      raise ValueError("learner Q didn't update target Q")
    # =============================================================================

  def build_model(self, input_shape, 
                        output_size,     
                        double_q,       
                        dueling_q,       
                        discount_factor):
    
    m = ModelBuilder()

    def modeling(x, scope):
      with tf.variable_scope(scope):
        layer       = m.preprocess(normalize=True)(x) 
        layer       = m.conv2d(32, 8, 4, tf.nn.relu)(layer)
        layer       = m.conv2d(64, 4, 2, tf.nn.relu)(layer)
        layer       = m.conv2d(64, 3, 1, tf.nn.relu)(layer)
        layer       = m.flatten(layer)

        if dueling_q:
          value_fc  = m.dense(512, tf.nn.relu)(layer)
          value     = m.dense(1)(value_fc)
          policy_fc = m.dense(512, tf.nn.relu)(layer)
          policy    = m.dense(output_size)(policy_fc)

          policy_mean = tf.reduce_mean(policy, axis=1, keepdims=True)
          q         = value + (policy - policy_mean)

        else:
          fc        = m.dense(512, tf.nn.relu)(layer)
          q         = m.dense(output_size)(fc)
        
        return q

    # Learner DQN
    # =============================================================================
    self.inputs = tf.placeholder(tf.float32, [None, *input_shape], 'states')
    self.q = modeling(self.inputs, self.learn_dqn_scope)
    self.max_q = tf.reduce_max(self.q, axis=1)
    self.argmax_q = tf.argmax(self.q, axis=1)
    self.model_layers = m.layers
    # =============================================================================

    # Target DQN
    # =============================================================================
    self.t_inputs = tf.placeholder(tf.float32, [None, *input_shape], 't_states')
    self.t_q = modeling(self.t_inputs, self.target_dqn_scope)
    self.t_max_q = tf.reduce_max(self.t_q, axis=1)
    self.t_argmax_q = tf.argmax(self.t_q, axis=1)
    # =============================================================================

    self.actions    = tf.placeholder(tf.int32, [None], name='actions')
    self.rewards    = tf.placeholder(tf.float32, [None], name='rewards')
    self.dones      = tf.placeholder(tf.int32, [None], name='dones')

    if double_q: 
      
      q_one_hot = tf.one_hot(self.argmax_q, output_size)
      target_q  = tf.reduce_sum(self.t_q * q_one_hot, axis=1)

      # (done) target = (learner DQN이 행동한) target_Q * γ + reward
      # (not done) target = reward
      target = tf.cast(1-self.dones, tf.float32) * target_q * discount_factor \
              + self.rewards

    else:
      # (done) target = max(target_Q) * γ + reward
      # (not done) target = reward
      target = tf.cast(1-self.dones, tf.float32) * self.t_max_q * discount_factor \
             + self.rewards

    # 행동한 Q값
    action_one_hot = tf.one_hot(self.actions, output_size)
    acted_q        = tf.reduce_sum(self.q * action_one_hot, axis=1)
    
    # loss = (reward + targetQ * γ) - Q
    self.loss = m.loss_func("HUBER", acted_q, target)
    var_list = tf.trainable_variables(self.learn_dqn_scope)
    self.lr  = tf.train.exponential_decay(
                          learning_rate=0.00025,
                          global_step=tf.train.get_or_create_global_step(),
                          decay_steps=100000,
                          decay_rate=1.0,
                          staircase=True)

    self.train_op   = m.train_op(optim_type='Adam',
                                 loss=self.loss,
                                 learning_rate=self.lr,
                                 var_list=var_list,
                                 epsilon=0.01)

    
