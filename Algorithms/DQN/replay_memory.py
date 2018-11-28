import numpy as np

class ReplayMemory():
  def __init__(self, size):
    """ Replay Buffer. DQN의 Experience를 저장

    Parameters
    ----------
    size: int
      memory 크기로, 만약 초과시 가장 오래된 메모리를 지우고 추가한다.
    """
    self._states      = None
    self._actions     = None
    self._rewards     = None
    self._next_states = None
    self._deads       = None

    self._max_size   = size
    self._index      = 0
    self._cur_size   = 0

  def __len__(self):
    return self._cur_size
    
  def save(self, state, action, reward, next_state, dead):
    ''' Memory에 상태, 행동, 보상, 다음 상태, 게임 종료 여부 저장 
    
    Parameters
    ----------
    state: np.array, uint8
      행동이 취해지기 전 상태
    action: np.array, uint8
      상태에 따른 행동
    reward: np.array, int8
      행동에 따른 보상
    next_state: np.array, uint8
      행동에 따른 다음 상태
    dead: np.array, bool
      죽었는지 여부
    
    '''
    state = state
    next_state = next_state
    
    if self._cur_size == 0:
      self._states      = np.zeros([self._max_size] + list(np.shape(state)), 
                                    dtype=np.uint8)
      self._actions     = np.zeros([self._max_size], 
                                    dtype=np.uint8)
      self._rewards     = np.zeros([self._max_size], 
                                    dtype=np.int8)
      self._next_states = np.zeros([self._max_size] + list(np.shape(next_state)), 
                                    dtype=np.uint8)
      self._deads       = np.zeros([self._max_size], 
                                    dtype=np.bool)

    self._states[self._index]      = state
    self._actions[self._index]     = action
    self._rewards[self._index]     = reward
    self._next_states[self._index] = next_state
    self._deads[self._index]       = dead

    if self._cur_size < self._max_size:
      self._cur_size += 1

    self._index = (self._index + 1) % self._max_size
 
  def sample(self, batch_size):
    ''' 메모리에서 임의의 데이터를 배치 사이즈 만큼 가져온다.
    
    Parameters
    ----------
    batch_size: int
      메모리에서 가져올 데이터 크기

    Returns
    -------
    self.states: np.array, uint8
      행동이 취해지기 전 상태
    self.actions: np.array, uint8
      상태에 따른 행동
    self.rewards: np.array, int8
      행동에 따른 보상
    self.next_states: np.array, uint8
      행동에 따른 다음 상태
    self.deads: np.array, bool
      죽었는지 여부
    '''
    rand_indexes = np.random.randint(0, self._cur_size, batch_size)

    return self._states[rand_indexes], \
           self._actions[rand_indexes], \
           self._rewards[rand_indexes], \
           self._next_states[rand_indexes], \
           self._deads[rand_indexes]