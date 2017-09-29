import gym
import numpy as np
import random
import 

GAME_SELECT = 'SpaceInvaders-v0'
#Hyper-Parameters
EPSILON = 
GAMMA = 
NO_OF_ITERATIONS =  
class Agent:
	number_of_actions=0
	explored_count = 0
	def __init__(self, action_space):
		self.number_of_actions = action_space
		self.explored_count = 0
	def get_nextaction(self,state):
		


if __name__ == '__main__':
	env = gym.make(GAME_SELECT)
	state = env.reset()
	AI = Agent(env.action_space.n)
	for i in NO_OF_ITERATIONS:
		action = Agent.get_nextaction(state)
