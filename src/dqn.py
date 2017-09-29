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



LEARNING_RATE=



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

        