# ======================================================================
# Author : MCLearning2
# Date : 2018.10.01 ~ 
# Goal : 강화학습 Baseline의 코드들을 Open AI gym의 Atari에서 구현하기
# Implemented Agent : DQN, A3C
# Implemented Environment : gym
# Reference : dennybritz/reinforcement-learning
#            (https://github.com/dennybritz/reinforcement-learning)
# ======================================================================

import os
import tensorflow as tf

from Common import initialize

# Tensorflow 속도를 높이기 위한 알고리즘
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

# tensorflow log 없애기
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
  parser = initialize.get_parser()

  dirs = initialize.set_debug_dir(parser.project_name,
                                  parser.folder_set, 
                                  parser.debug_dir) 

  agent = initialize.select_agent(parser.agent, parser.environment, dirs)
  agent.learn()

if __name__ == '__main__':
  main()
  
  