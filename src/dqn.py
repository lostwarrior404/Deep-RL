import gym
import numpy as np
import random
from __future__ import print_function
import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
import sys
import random
import numpy as np
from collections import deque
from keras import initializers
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
import glob
import csv
import pickle
from sklearn.cluster import KMeans
from keras.models import load_model

LEARNING_RATE=1e-4
BATCH_SIZE=40
GAME_SELECT = 'SpaceInvaders-v0'
EPSILON = 0.2
CHANGE_EPSILON=0.00000000001
GAMMA = 0.95
NO_OF_ITERATIONS =  
MEMORY_SIZE = 1000
REPLAY_SIZE = 

def frame_process(state):
	state = skimage.color.rgb2gray(state)
    state = skimage.transform.resize(state,(84,84))
    state = skimage.exposure.rescale_intensity(state,out_range=(0,255))
    return state
    
def transition_process(transition):
	temp = []
	for i in range(REPLAY_SIZE):
		temp.append(transition[i][0])
	temp.append(transition[REPLAY_SIZE-1][1],transition[REPLAY_SIZE-1][2],transition[REPLAY_SIZE-1][3],transition[REPLAY_SIZE-1][4])
	return temp
	

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
		if EPSILON > FINAL_EPSILON and explored_count > OBSERVE:
		    	EPSILON -= CHANGE_EPSILON
		self.explored_count+=1
	def replay(self):
		if(explored_count>OBSERVE):
			mini_batch=random.sample(memory,BATCH_SIZE)
		self.logic.train(mini_batch)

class Model:
	number_of_actions=0
	explored_count = 0
	model=0
	input_shape=0
	
	def __init__(self, action_space):
		self.number_of_actions = action_space
		shape=(84,84,4)
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(8, 8),strides=4,activation='relu',input_shape=shape))
		model.add(Conv2D(64, kernel_size=(4, 4),strides=2,activation='relu'))
		model.add(Conv2D(64, kernel_size=(3, 3),strides=1,activation='relu'))
		model.add(Dense(512,activation='relu'))
		model.add(Dense(action_space,activation='softmax'))
		adam = Adam(lr=LEARNING_RATE)
		model.compile(loss='mse',optimizer=adam)
		self.model=model
	
	def predict(state):
		mreturn self.model.predict(state)

	def train(self,batch):
		inputs = np.zeros((len(batch), 84, 84,4))   #32, 80, 80, 4
		targets = np.zeros((inputs.shape[0], ACTIONS))     
		for i in range(0, len(batch)):
				state_t = minibatch[i][0]
				action_t = minibatch[i][1]
				reward_t = minibatch[i][2]
				state_t1 = minibatch[i][3]
				terminal = minibatch[i][4]
				inputs[i:i + 1] = state_t    #I saved down s_t

				targets[i] = model.predict(state_t)  # Hitting each buttom probability
				Q_sa = model.predict(state_t1)

				if terminal:
					targets[i, action_t] = reward_t
				else:
					targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
		 loss += model.train_on_batch(inputs, targets)

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

		
