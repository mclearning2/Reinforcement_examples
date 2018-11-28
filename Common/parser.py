from argparse import ArgumentParser

def _bool(s):
  # 일정 단어로 True False를 표현
  # parser는 bool을 str로 받아서 무조건 True가 나온다.
  # 따라서 bool 대신 임의의 함수로 type을 대신한다.(parser.register)
  return s.lower() in ("yes", "true", "t", "1")

class Parser():
  def __init__(self):
    self._parser = ArgumentParser()
    self._parser.register('type', 'bool', _bool)
  
  def add(self, name, default, help=None):
    assert isinstance(name, str)    
    
    data_type = type(default)
    if data_type == bool:
      data_type = _bool

    self._parser.add_argument('--' + name, 
                              type=data_type, 
                              default=default, 
                              help=help)
  
  def parse_args(self):
    return self._parser.parse_args()