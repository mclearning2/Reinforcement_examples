
import tensorflow as tf
from Common.modelbuilder import ModelBuilder

class A3CNetwork:
  ''' A3C를 위한 Network. Global Network와 local Network 모두 이 클래스 사용

    Parameters
    ----------
    input_shape: list or tuple
      상태 shape [height, width, channels]
    
    output_size: int
      행동 가짓수

    name: str
      모델의 scope name
  '''
  def __init__(self, input_shape, output_size, scope_name, global_scope = None):
    
    self.scope_name = scope_name
    self.input_shape = input_shape
    self.output_size = output_size
    self.model_layers = None

    self.build_model()
    if global_scope:
      self.build_train(global_scope = global_scope)

  def eval_policy(self, sess, state):
    return sess.run(self.policy, feed_dict={self.inputs:state})
  
  def eval_value(self, sess, state):
    return sess.run(self.value, feed_dict={self.inputs:state})

  def build_model(self):
    m = ModelBuilder()
    self.model_layers  = m.layers

    with tf.variable_scope(self.scope_name):
      self.inputs = tf.placeholder(tf.float32, [None, *self.input_shape])
      layer = m.preprocess(normalize=True)(self.inputs)
      layer = m.conv2d(32, 5, 3, tf.nn.relu)(layer)
      layer = m.conv2d(64, 4, 2, tf.nn.relu)(layer)
      layer = m.conv2d(64, 3, 1, tf.nn.relu)(layer)
      layer = m.flatten(layer)
      layer = m.dense(512, tf.nn.relu)(layer)
      self.policy = m.dense(self.output_size, tf.nn.softmax)(layer)
      self.value  = m.dense(1, None)(layer)


  def build_train(self, global_scope):
    # Actor Loss
    # =====================================================================================
    self.actions      = tf.placeholder(shape=[None], dtype=tf.int32)
    self.target_value = tf.placeholder(shape=[None], dtype=tf.float32)
    self.advantages   = tf.placeholder(shape=[None], dtype=tf.float32)

    action_onehot     = tf.one_hot(self.actions, self.output_size, dtype=tf.float32)
    action_prob       = tf.reduce_sum(self.policy * action_onehot, [1])

    entropy           = -tf.reduce_sum(self.policy * tf.log(self.policy + 1e-6))
    
    self.policy_loss  = -tf.reduce_sum(tf.log(action_prob + 1e-6) * self.advantages)

    td_error          = self.target_value - tf.reshape(self.value, [-1])
    self.value_loss   = 0.5 * tf.reduce_sum(tf.square(td_error))

    self.loss         = 0.5 * self.value_loss + self.policy_loss - entropy * 0.01

    local_vars        = tf.trainable_variables(self.scope_name)
    global_vars       = tf.trainable_variables(global_scope)

    gradients         = tf.gradients(self.loss, local_vars)
    grads, _          = tf.clip_by_global_norm(gradients, 40.0)

    self.lr           = 0.0001
    optimizer         = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-6)
    self.train_op     = optimizer.apply_gradients(zip(grads,global_vars), 
                                                  global_step=tf.train.get_or_create_global_step())

  
