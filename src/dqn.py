import gym
import numpy as np
import random
import 

GAME_SELECT = 'SpaceInvaders-v0'
#Hyper-Parameters
EPSILON = 
GAMMA = 
NO_OF_ITERATIONS =  
MEMORY_SIZE = 
REPLAY_SIZE = 
def frame_process(state):


class Agent:
	number_of_actions=0
	explored_count = 0
	memory = []
	def __init__(self, action_space):
		self.number_of_actions = action_space
		self.explored_count = 0
		self.logic = Model(number_of_actions)

	def get_nextaction(self,state):
		if(random.random()<EPSILON):
			return random.randint(0,self.number_of_actions-1)
		else:
			return numpy.argmax(logic.predict(state))	

	def record(self,observation):
		if(len(memory)==MEMORY_SIZE):
			self.memory.pop(0)
		self.memory.append(observation)

		self.explored_count+=1





if __name__ == '__main__':
	env = gym.make(GAME_SELECT)
	state = env.reset()
	AI = Agent(env.action_space.n)
	for i in xrange(NO_OF_ITERATIONS):
		replay_memory = []
		for j in xrange(REPLAY_SIZE):
			env.render()
			action = AI.get_nextaction(state)
			next_state, reward, done, info = env.step(action)
			state = frame_process(state)
			next_state = frame_process(next_state)
			replay_memory.append([state,action,reward,next_state])
		AI.record(replay_memory)
	if done:
            print("Episode finished after {} timesteps".format(i+1))
            break

        