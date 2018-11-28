import os
import gym
from argparse import ArgumentParser

from Common import utils

def _bool(s):
  # 일정 단어로 True False를 표현
  # parser는 bool을 str로 받기 때문에 무조건 True가 나온다.
  # 따라서 bool 대신 임의의 함수로 type을 대신한다.(parser.register)
  return s.lower() in ("yes", "true", "t", "1")

def get_parser():
  ''' 실행시킬 때 받을 parameter 설정(parser) '''
  _parser = ArgumentParser()
  _parser.register('type', 'bool', _bool)

  _parser.add_argument('-A', '--agent', type=str,
                      default='A3C',
                      help="[ 사용하길 원하는 Agent ]\n"
                           "사용가능 : DQN, A3C")

  _parser.add_argument('-E', '--environment', type=str,
                      default='BreakoutDeterministic-v4',
                      help="[ 학습하길 원하는 환경 ]\n"
                      "Open AI Gym (참조 : https://gym.openai.com/envs)")

  _parser.add_argument("-P", '--project_name', type=str,
                      default = None,
                      help="학습 데이터들을 저장하거나 불러올 폴더 이름")

  _parser.add_argument("-F", '--folder_set', type=str,
                      default = 'make',
                      help="디버그용으로 생성하는 폴더들을 다룸.\n"
                           "-delete : 기존 것이 존재하면 지움\n"
                           "-make : 기존 것이 존재하면 index를 추가해서 만듦\n"
                           "-restore : 기존 것이 존재하면 그대로 사용해서 복원\n")

  _parser.add_argument("-D", '--debug_dir', type=str,
                      default = './Debug',
                      help="디버그용 폴더들을 저장할 상위 폴더")

  parser = _parser.parse_args()  

  # Agent Check
  # ===============================================================================================
  assert parser.agent in ['DQN', 'A3C'], \
         f"다음 Agent 중 하나를 선택해야 합니다.['DQN', 'A3C'], '{parser.agent}'은 유효하지 않습니다.."
  # ===============================================================================================

  # Environment Check
  # ===============================================================================================
  env_exist = False
  for env_spec in gym.envs.registry.all():
    if env_spec.id == parser.environment:
      env_exist = True

  assert env_exist, "올바른 Environment를 골라야 합니다. 다음 사이트를 참조해주세요" \
                    "(https://gym.openai.com/envs/#atari)"  

  # ===============================================================================================

  # Folder set check
  # ===============================================================================================
  assert parser.folder_set in ['make', 'restore', 'delete'], \
         "'delete', 'make' 또는 'restore' 중 하나를 선택해야합니다. " + \
        f"'{parser.folder_set}'은 유효하지 않습니다."
  # ===============================================================================================


  if parser.project_name is None:
    parser.project_name = parser.agent + '_' + parser.environment

  

  return parser

def set_debug_dir(project_name, folder_set, debug_dir='Debug/'):
  ''' 프로젝트에 대한 정보를 기록할 폴더들을 생성(summary, checkpoint, logging, video)
  
  Parameters
  ----------
  project_name: str
    프로젝트 이름으로 기록할 폴더들에 이 이름이 붙여진다.

  folder_set: str
    'delete', 'make', 'restore' 세가지 목적 중 하나로 폴더를 다룸

  debug_dir: str
    디버그를 위한 폴더들을 저장할 폴더 위치
  '''

  dirs = {'summary':None, 
          'checkpoint': None, 
          'video': None,
          'logs': None}

  for d in dirs:
    dirs[d] = os.path.join(debug_dir, d, project_name)
  
  # =======================================================================
  if folder_set == 'delete':
    try:
      for d in dirs:
        utils.remove_dir(dirs[d])
    except OSError:
      raise FileExistsError("폴더의 접근 권한이 제한되어있습니다."
                            "Tensorboard이 실행되어있는지, "
                            "이전 프로그램이 실행되고 있는 지 확인해주세요.")
  # =======================================================================
  elif folder_set == 'make':
    index = 1
    # 폴더 하나라도 존재하면 ...
    while True:

      all_not_exist = True
      for d in dirs:
        if os.path.exists(dirs[d]):
          all_not_exist = False

      if all_not_exist:
        break

      prj_name       = project_name + f"_{index}"
      for d in dirs:
        dirs[d] = os.path.join(debug_dir, d, prj_name)
      index += 1
  # =======================================================================
  elif folder_set == 'restore':
    pass
  # =======================================================================


  if folder_set != 'restore':
    for d, path in dirs.items():
      print(f"{d} will be saved in '{path}'")
      utils.make_dir(path)    


  return dirs

def select_agent(agent_name, env_name, dirs):
  ''' Agent를 선택

  Parameters
  ---------
  agent_name : str
    에이전트 이름 ex. DQN, A3C
  
  env_name : str
    환경 이름 ex. BreakoutDeterministic-v4
  
  dirs : dict
    debugging을 위한 폴더 경로들 ex. logs, checkpoint
  '''

  if 'DQN' == agent_name:
    from Algorithms.DQN.agent import Agent    

  elif 'A3C' == agent_name:
    from Algorithms.A3C.agent import Agent  

  agent = Agent(env_name, dirs)

  return agent



