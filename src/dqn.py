from __future__ import print_function
import gym
import numpy as np
import random
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
from keras.layers.convolutional import Conv2D
import tensorflow as tf
import glob
import csv
import pickle
from sklearn.cluster import KMeans
from keras.models import load_model

LEARNING_RATE=1e-4
BATCH_SIZE=40
GAME_SELECT = 'SpaceInvaders-v0'
INITIAL_EPSILON=?????
EPSILON = 0.2
CHANGE_EPSILON=0.00000000001
GAMMA = 0.95
NO_OF_ITERATIONS =  10000000
REPLAY_MEMORY_SIZE = 10000
OBSERVE = 320
FINAL_EPSILON = 0.001

def frame_process(state):
	state = skimage.color.rgb2gray(state)
	im = state[25:197,:]
	state = skimage.transform.resize(im,(84,84))
	state = skimage.exposure.rescale_intensity(state,out_range=(0,255))
	return state

class Agent:
	number_of_actions=0
	explored_count = 0
	memory = []
	def __init__(self, action_space):
		self.number_of_actions = action_space
		self.explored_count = 0
		self.logic = Model(self.number_of_actions)

	def get_nextaction(self,state):
		if(random.random()<EPSILON):
			return random.randint(0,self.number_of_actions-1)
		else:
			x=self.logic.predict(state)
			return np.argmax(x)   

	def record(self,observation):
		global EPSILON
		if(len(self.memory)>=REPLAY_MEMORY_SIZE):
			self.memory.pop(0)
		self.memory.append(observation)

		if EPSILON > FINAL_EPSILON and self.explored_count > OBSERVE:
			EPSILON = max(FINAL_EPSILON, INITIAL_EPSILON - (INITIAL_EPSILON- FINAL_EPSILON) * self.explored_count/100000000)
			EPSILON = max(.1, 1.0 - 0.9 * self.explored_count / 1e7)
			# EPSILON -= CHANGE_EPSILON
		self.explored_count+=1
	def replay(self):
		if(self.explored_count>OBSERVE):
			mini_batch=random.sample(self.memory,BATCH_SIZE)
			self.logic.train(mini_batch)
	def saveit(self):
		self.logic.saveit()

class Model:
	explored_count = 0
	input_shape=0
	
	def __init__(self, action_space):
		self.number_of_actions = action_space
		shape=(84,84,4)
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(8, 8),strides=4,activation='relu',input_shape=shape))
		model.add(Conv2D(64, kernel_size=(4, 4),strides=2,activation='relu'))
		model.add(Conv2D(64, kernel_size=(3, 3),strides=1,activation='relu'))
		model.add(Flatten())
		model.add(Dense(512,activation='relu'))
		model.add(Dense(action_space,activation='softmax'))
		adam = Adam(lr=LEARNING_RATE)
		model.compile(loss='mse',optimizer=adam)
		self.model=model
		self.loss=0
	
	def predict(self,state):
		return self.model.predict(state)

	def saveit(self):
		self.model.save('atari-model.h5',overwrite=True)

	def train(self,batch):
		inputs = np.zeros((len(batch), 84, 84,4))   #32, 80, 80, 4
		targets = np.zeros((inputs.shape[0], self.number_of_actions))     
		for i in range(0, len(batch)):
				state_t = batch[i][0]
				action_t = batch[i][1]
				reward_t = batch[i][2]
				state_t1 = batch[i][3]
				terminal = batch[i][4]
				inputs[i:i + 1] = state_t    #I saved down s_t

				targets[i] = self.model.predict(state_t)  # Hitting each buttom probability
				Q_sa = self.model.predict(state_t1)

				if terminal:
					targets[i, action_t] = reward_t
				else:
					targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
		self.loss += self.model.train_on_batch(inputs, targets)

if __name__ == '__main__':
	env = gym.make(GAME_SELECT)
	state = env.reset()
	state = frame_process(state)
	state_arr = np.stack((state, state, state, state), axis=-1)
	#keras Requirement for reshape
	state_arr = state_arr.reshape(1,state_arr.shape[0],state_arr.shape[1],state_arr.shape[2])
	AI = Agent(env.action_space.n)
	for i in xrange(NO_OF_ITERATIONS):
		env.render()
		action = AI.get_nextaction(state_arr)
		print(action)
		next_state, reward, done, info = env.step(action)
		next_state = frame_process(next_state)
		next_state = next_state.reshape(1,next_state.shape[0],next_state.shape[1],1)
		next_state_arr = np.append(next_state,state_arr[:,:,:,:3],axis=3)
		AI.record([state_arr,action,reward,next_state_arr,done])
		AI.replay()
		state_arr=next_state_arr
		print(i)
		if(i%1000==0):
			AI.saveit()

		if done:
			print("Episode finished after {} timesteps".format(i+1))
			state = env.reset()
			state = frame_process(state)
			state_arr = np.stack((state, state, state, state), axis=-1)
			state_arr = state_arr.reshape(1,state_arr.shape[0],state_arr.shape[1],state_arr.shape[2])
