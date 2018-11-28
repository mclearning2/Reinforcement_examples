import gym
import random
import numpy as np
from gym.wrappers import Monitor

class Atari:
  ''' Open AI gym의 Atari 2600 게임 환경

    Parameters
    ----------
      env_id: str
        gym의 환경 id (참조 : https://gym.openai.com/envs)

      video_dir: str
        monitor를 할 폴더 이름

      width: int
        상태를 전처리해서 되길 원하는 폭

      height: int
        상태를 전처리해서 되길 원하는 높이

      n_history: int
        history 개수를 결정

      no_op: int
        처음에 랜덤한 시작을 위한 step 범위(1 ~ no_op)
    '''
  def __init__(self, env_id = 'BreakoutDeterministic-v4',
                     n_history = 4,
                     no_op = 30):

    assert isinstance(env_id, str)
    assert isinstance(n_history, int)
    assert isinstance(no_op, int)

    self.name          = env_id
    self.no_op         = no_op   
 
    self.env           = gym.make(self.name)
    self.valid_actions = self.env.unwrapped.get_action_meanings()

    tmp_image          = np.zeros(self.env.observation_space.shape)
    width, height      = self.preprocess(tmp_image).shape
    self.n_history     = n_history 
    self.state_size    = [width, height, n_history]
    self.action_size   = self.env.action_space.n

  def __del__(self):
    ''' 환경 종료 '''
    self.env.close()

  def monitor(self, video_dir, resume=True, force=False, 
                    video_callable = lambda count: count % 50 == 0):
    ''' 주기적으로 비디오를 통해 agent 학습하는 영상 저장 
    
    Parameters
    ----------
    video_dir: str
      비디오(mp4)를 저장할 폴더
    resume: bool
      이어서 저장할지 여부
    force: bool
      기존 것을 지울지 여부
    callable_rate: int
      video 저장 주기(episode 마다)
    '''
    self.env = Monitor(self.env, video_dir, resume=resume, force=force,
                       video_callable=video_callable)
  
  def random_action(self):
    ''' 임의의 행동을 반환 '''
    return self.env.action_space.sample()

  def preprocess(self, image):
    ''' 이미지 크기를 가로세로 반으로 줄이고 grayscale로 변화시킨다.
    Parameters
    ----------
      image: np.array, np.int32
        0~255범위 값의 칼라 이미지(210, 160, 3)
    Returns
    -------
      image: np.array, uint8
        전처리된 이미지 (105, 80)
    '''

    image = image[:, :, 0] * 0.2989 + \
            image[:, :, 1] * 0.5870 + \
            image[:, :, 2] * 0.1140
    preprocessed = image[::2, ::2]

    return preprocessed

  def reset(self):
    ''' 게임을 새로 시작 '''
    state        = self.env.reset()
    self.score   = 0
    self.steps   = 0

    for _ in range(np.random.randint(1,self.no_op)):
      state, _, _, info = self.env.step(1)

    if 'ale.lives' in info:
      self.life_game = True
      self.lives = info['ale.lives']
    else:
      self.life_game = False

    preprocessed = self.preprocess(state)
    self.history = np.stack([preprocessed] * self.n_history, axis=2)

    return self.history
    
  def step(self, action):
    ''' action을 통해서 환경을 변화시킨다.

    Parameters
    ----------
      action: int
        [0, self.action_size) 크기의 action. breakout 경우 0 ~ 3
    
    Returns
    -------
      next_history: np.array uint8
        다음 상태
      reward: int
        -1 ~ 1 범위의 보상
      done: bool
        한 게임이 종료되었는지 여부
      dead: bool
        죽었는지 여부
    '''

    next_state, reward, done, info = self.env.step(action)

    next_state   = self.preprocess(next_state)
    next_state   = np.expand_dims(next_state, axis=2)

    next_history = np.concatenate((self.history[:, :, 1:], next_state), axis=2)

    self.history = next_history

    self.steps   += 1
    self.score   += reward

    reward = np.clip(reward, -1, 1)
    
    if self.life_game:
      if self.lives > info['ale.lives']:
        dead = True
        self.lives = info['ale.lives']
      else:
        dead = False
    else:
      dead = done   

    return self.history, reward, done, dead
    
  def render(self):
    self.env.render()
