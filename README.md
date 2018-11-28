## 목표
- 기본적인 RL 알고리즘 모델 구현해보기
- 범용적인 목적보다는 구현해보는 경험을 우선순위로 하기 때문에 환경은 OpenAI GYM Atari로 제한.(후에 추가할지는 고민...)

## 실행 환경
저는 아래와 같은 환경에서 코드를 작성했기 때문에 이외의 환경으로 인한 문제는...
- Windows 10
- Python 3.6.6
- Tensorflow 1.11.0
- CUDA 9.0
- cudnn 7.2.1
- OpenAI gym Atari games

## 실행방법

### 코드 다운로드
```
git clone https://github.com/mclearning2/Reinforcement_examples.git
```

### 필요한 패키지 설치
```
pip install -r requirements.txt
```

Atari 환경은 별개로 설치 (윈도우용으로)
```
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
```

### 실행하기

```
cd Reinforement_examples
python main.py -A [Agent name] -E [Atari game id] -P [Project name] -F [Folder setting] -D [Debug dir]
```
#### Argument

|  Arg 	| Help 	|
| :---: | :-	|
|  -A		|  (--agent) Environment와 상호작용할 Agent 이름. 아래 Algorthms를 참조. **[Default : DQN]** |
|  -E 	| (--environment) Atari 게임 이름. [OpenAI gym Environment](https://gym.openai.com/envs/#atari)에서 확인할 수 있습니다.  **[Default : BreakoutDeterministic-v4]**|
|  -P 	| (--project_name) 프로젝트 이름. summary, checkpoint, logging 등 디버깅용으로 저장할 폴더 이름. 정하지 않을 경우 Agent이름과 Environment이름을 합친 것이 프로젝트 이름이 된다. 예) DQN과 Breakout-v0이라면, DQN_Breakout-v0 **[Default : None]**|
|  -F		| (--folder_set) mode가 train일 때만 적용. 위 project name에 의해서 폴더를 생성하려 할 떄 동일한 이름이 있는 경우 다음과 같이 처리. delete(기존 폴더 삭제) / make(index를 추가해서 새로 생성. 예. project_name_1) / restore(모델 복원). **[Default : make]**|
|  -D		| (--debug_dir) 디버깅의 상위 폴더 이름 **[Default : debug]**|

#### 예
- Agent : DQN
- Environment : BreakoutDeterministic-v4
- Project name : Test1
- Folder setting : delete
- Debug directory : debug

```
python main.py -A DQN -E BreakoutDeterministic-v4 -P Test1 -F delete -D debug
```

## 중간 결과 확인하기
Debug directory에 4개의 폴더가 생성된다. 

- checkpoint : Agent의 모델이 저장
- logs : command outputs들의 log 저장
- summary : Tensorboard의 scalars
- video : 중간중간 테스트를 위해 플레이한 Atari video 파일(.mp4)

그리고 그 내부에는 각각 Project name으로 만들어진 폴더가 있다. 만약 tensorboard를 확인하려면 다음과 같이 하면된다.

```
tensorboard --logdir=[Debug 폴더명]\summary
```

만약 가중치를 복원해서 학습하고 싶다면 folder_set에서 restore를 사용한다. 하지만 memory 저장은 안했기 때문에 학습을 다시하는 게 잘 안되므로 (앞으로 구현해야하겠지만) 안하는 것을 추천

## Algorithms

- [DQN (Deep Q Network)](https://github.com/mclearning2/Reinforement_examples/tree/master/Algorithms/DQN)
- A3C (Asynchronous Actor-Critic Agents) (진행중)

## Reference
- https://github.com/dennybritz/reinforcement-learning