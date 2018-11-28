import numpy as np
import tensorflow as tf

class Summary:
  ''' 모델의 log를 기록하는 클래스 '''
  def __init__(self, summary_dir, sess, scope_name):
    self._writer = tf.summary.FileWriter(summary_dir, sess.graph)
    self._summary_ph = dict()
    self._summary = dict()
    self._scope_name = scope_name

  def write(self, sess, step, summary_dict):
    ''' summary 값들을 모아서 writer로 저장.
    
    Parameters
    ----------
    sess: tf.Session()

    step: int
      summary의 x축에 해당하는 step

    scope_name: str
      tensorboard에서 저장될 summary들의 scope name

    summary_dict: dict
      summarys에 기록할 딕셔너리 {summary keys : value}
    
    '''
    try:
      selected_summaries = list()
      feed_dict          = dict() 
      with tf.variable_scope(self._scope_name):
        for name, value in summary_dict.items():
          if name not in self._summary.keys():
            self._summary_ph[name] = tf.placeholder(np.array(value).dtype, name=name)
            self._summary[name] = tf.summary.scalar(name, self._summary_ph[name])

          selected_summaries.append(self._summary[name])
          feed_dict[self._summary_ph[name]] = value
    
      merged_summary = tf.summary.merge(selected_summaries)
      summarys = sess.run(merged_summary, feed_dict=feed_dict)

      self._writer.add_summary(summarys, global_step=step)
    except ValueError:
      raise ValueError("'%s' is not a valid scope name. 띄어쓰기 같은거..." % name)