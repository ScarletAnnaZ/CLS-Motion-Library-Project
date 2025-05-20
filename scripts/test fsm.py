# test_fsm.py

from fsm_agent import AgentFSM

agent = AgentFSM()
print("初始状态：", agent.current_state())  # 应该是 idle

agent.got_follow()
print("got_elephant (human subject)  后状态：", agent.current_state())  # 应该是 follow

agent.got_highlight()
print("got_salsa dance  后状态：", agent.current_state())  # 应该是 highlight

agent.got_dodge()
print("got_nursery rhyme - Cock Robin 后状态：", agent.current_state())  # 应该是 dodge

agent.got_idle()
print("got_walking, running, kicking, punching, knee kicking, and stretching 后状态：", agent.current_state())  
