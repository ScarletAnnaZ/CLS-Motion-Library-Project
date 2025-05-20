# fsm_agent.py

from transitions import Machine

# 1) 定义所有可能的状态
states = ['salsa dance', 'walking, running, kicking, punching, knee kicking, and stretching', 'nursery rhyme - Cock Robin', 'elephant (human subject)']

# 2) 定义从状态 A，经由事件 E，去状态 B 的规则
transitions = [
    # 当收到“follow”标签时，从任何状态跳到 follow
    {'trigger': 'got_salsa dance',    'source': '*',          'dest': 'salsa dance'},
    # 收到“highlight”标签时，从任何状态跳到 highlight
    {'trigger': 'got_nursery rhyme - Cock Robin', 'source': '*',          'dest': 'nursery rhyme - Cock Robin'},
    # 收到“dodge”标签时，从任何状态跳到 dodge
    {'trigger': 'got_walking, running, kicking, punching, knee kicking, and stretching',     'source': '*',          'dest': 'walking, running, kicking, punching, knee kicking, and stretching'},
    # 收到“idle”（或未识别标签）时，回到 idle
    {'trigger': 'got_elephant (human subject)',      'source': '*',          'dest': 'salsa dance'},
]

class AgentFSM:
    def __init__(self):
        # 初始化状态机，默认状态 idle
        self.machine = Machine(model=self, states=states, transitions=transitions, initial='salsa dance')

    def current_state(self):
        return self.state
