from __future__ import annotations
import random

from game import Game,State
from collections import deque
from typing import Tuple
import numpy as np
import keras
import tensorflow as tf
import os.path
class Agent:
    def __init__(self,state_shape=None,action_dim=None,filename=None):
        self.model  = None
        if filename and os.path.isfile(filename):
            self.model = keras.models.load_model(filename)
            return
        self.generate(state_shape,action_dim)
    def generate(self,state_shape,action_dim):
        learning_rate = 0.001
        init = tf.keras.initializers.HeUniform()
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(action_dim, activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
        self.model= model
    def predict(self,states):
        return self.model.predict(np.array([x.feed for x in states]),verbose=1)
    def copy(self,a:Agent):
        self.model.set_weights(a.model.get_weights())
    def fit(self,states,nactions,batch_size):
        self.model.fit(np.array([x.feed for x in states]),np.array(nactions),shuffle=True,verbose=0,batch_size=batch_size)
    def decide(self,state)-> Tuple[int,np.ndarray]:
        prediction = self.predict([state])[0]
        return np.argmax(prediction),prediction
    def save(self,filename):
        self.model.save(filename)
class Memory:
    def __init__(self,old_state:State,naction,new_state:State,reward:int,is_done:bool):
        self.old_state = old_state
        self.is_done = is_done
        self.naction = naction
        self.new_state = new_state
        self.reward = reward
    def __repr__(self):
        return f"<{self.old_state} {self.naction} {self.new_state} {self.reward}>"
def train(side_model:Agent,train_model:Agent,memories:deque[Memory],learning_rate=.7):
    discount = .618
    lr = learning_rate
    x=[]
    y=[]
    if len(memories)<128:
        return
    batch = random.sample(memories,128)
    f_qtable = side_model.predict([x.new_state for x in batch])
    c_qtable = train_model.predict([x.old_state for x in batch])
    for i, memory in enumerate(batch):
        if memory.is_done:
            future_q = memory.reward
        else:
            future_q = discount*np.max(f_qtable[i])+memory.reward
        c_qtable[i][memory.naction] = (1-lr)*c_qtable[i][memory.naction] + lr*future_q
        y.append(c_qtable[i])
        x.append(memory.old_state)
    print("training...")
    train_model.fit(x,y,128)


def play():
    dimensions = 10
    game = Game()
    game.generate(dimensions)
    train_model = Agent((dimensions*dimensions*4+4,),4)
    side_model = Agent((dimensions*dimensions*4+4,),4)
    side_model.copy(train_model)
    actions = ((0, 1), (1, 0), (-1, 0), (0, -1))
    game.generate(10)
    memory = deque(maxlen=50000)
    counter = 0
    counter_since_last_reset = 0
    while True:
        cur_state = game.state()
        naction, nparr = train_model.decide(cur_state)
        best_dir = actions[naction]
        new_state, reward,is_end = game.move(best_dir)
        memory.append(Memory(cur_state,naction,new_state,reward,is_end))
        print(game)
        if reward==100:
            print("CONGRAUTLATION1!!")
            game.generate(10)
        if counter%4==0:
            train(side_model,train_model,memory)

        if counter%100==0:
            side_model.copy(train_model)
        counter+=1
