import os
import pickle
import random
import threading
import time
import traceback
from typing import List
import cv2
import numpy as np
import pynput
from pynput.keyboard import Key, Listener
from config import downwell_file
import keras.layers
import win32com.client

from screenrecorder import *
from NN import *


class DAgent(Agent):
    def generate(self,state_shape,action_dim):
        inputs = keras.layers.Input(shape=state_shape)

        # Convolutions on the frames on the screen
        layer1 = keras.layers.Conv1D(32, 8, strides=4, activation="relu")(inputs)
        layer2 =  keras.layers.Conv1D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = keras.layers.Conv1D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = keras.layers.Flatten()(layer3)

        layer5 = keras.layers.LeakyReLU(512)(layer4)

        layer6 = keras.layers.LeakyReLU(256)(layer5)
        layer7 = keras.layers.LeakyReLU(128)(layer6)
        action = keras.layers.Dense(action_dim, activation="linear")(layer7)

        self.model = keras.Model(inputs=inputs, outputs=action)
        self.model.compile(keras.optimizers.Adam(learning_rate=0.00025,clipnorm=1.0),keras.losses.Huber())

class DoubleDAgent:
    def __init__(self,state_shape=None,action_dim=None,file=None):
        self.models = [DAgent(state_shape,action_dim,"a_"+file),DAgent(state_shape,action_dim,"b_"+file)]
        # https://nestedsoftware.com/2019/07/25/tic-tac-toe-with-tabular-q-learning-1kdn.139811.html
        self.discount = .618
        self.do_n_step = True
    def double_train(self, memories, model_index:int,lr=.7, batch_size=128):
        discount = self.discount
        x = []
        y = []
        if len(memories) < batch_size:
            return
        batch = random.sample(memories, batch_size)
        train_model = self.models[model_index]
        eval_model = self.models[model_index ^ 1]

        star_results = np.argmax(train_model.predict([x.new_state for x in batch]), axis=1)
        eval_results = eval_model.predict([x.new_state for x in batch])
        eval_action_rewards = eval_results[np.arange(len(eval_results)), star_results]

        c_qtable = train_model.predict([x.old_state for x in batch])
        for i, memory in enumerate(batch):
            if memory.is_done:
                future_q = memory.reward
            else:
                future_q = discount * eval_action_rewards[i] + memory.reward
            c_qtable[i][memory.naction] = (1 - lr) * c_qtable[i][memory.naction] + lr * future_q
            y.append(c_qtable[i])
            x.append(memory.old_state)
        print("training...")
        train_model.fit(x, y, batch_size)
    def decide(self,state:DState) -> Tuple[int,np.ndarray]:
        prediction = self.predict([state])[0]
        return np.argmax(prediction),prediction
    def predict(self,states:List[DState]):
        p1 = (self.models[0].predict(states))
        # p2 = (self.models[1].predict(states))
        # if len(states)==1:
        #     actions = ["left", "right", "jump", "jleft", "jright"]
        #     print("expected: ", list(zip(actions, map(int,p1[0]))))
        #     print("expected: ", list(zip(actions, map(int,p2[0]))))
        #     print()
        return p1
    def save(self,file):
        self.models[0].save("a_"+file)
        self.models[1].save("b_"+file)

class RDState(DState):
    def __init__(self, states):
        self.states = states
        self.feed = np.array([x.feed for x in states])
class RDAgent(DAgent):
    def generate(self,state_shape,action_dim):
        inputs = keras.layers.Input(shape=state_shape)

        # Convolutions on the frames on the screen
        layer1 = keras.layers.ConvLSTM1D(32, 8, strides=4, activation="relu",return_sequences=True)(inputs)
        l = keras.layers.MaxPool2D()(layer1)
        l = keras.layers.BatchNormalization()(l)
        l=  keras.layers.ConvLSTM2D(64, 4, strides=2, activation="relu",return_sequences=True)(l)
        l = keras.layers.MaxPool2D()(l)
        l = keras.layers.BatchNormalization()(l)
        l = keras.layers.Conv1D(64, 3, strides=1, activation="relu",return_sequences=False)(l)
        l = keras.layers.MaxPool1D()(l)
        l=keras.layers.BatchNormalization()(l)
        layer4 = keras.layers.Flatten()(l)

        layer5 = keras.layers.LeakyReLU(512)(layer4)

        layer6 = keras.layers.LeakyReLU(256)(layer5)
        layer7 = keras.layers.LeakyReLU(128)(layer6)
        action = keras.layers.Dense(action_dim, activation="linear")(layer7)

        self.model = keras.Model(inputs=inputs, outputs=action)
        self.model.compile(keras.optimizers.Adam(learning_rate=0.00025,clipnorm=1.0),keras.losses.Huber())
    def fit(self,states:RDState,rewards,batch_size):
        self.model.fit(np.array([x.feed for x in states]),np.array(rewards),shuffle=True,verbose=0,batch_size=batch_size)


def play_downwell_rnn():
    # shell = win32com.client.Dispatch("WScript.Shell")
    # os.startfile(downwell_file)
    # input("navigate to the point where you are actualy in the game and press enter")
    n_samples = 0
    memory = deque(maxlen=10000)
    sleep(1)
    game = DGame(.001,None)

    actions = ["left","right","jump","jleft","jright"]
    rnn_model=RDAgent(game.screen_dimensions+(n_samples,),len(actions),"downwellmodel_rnn.h5")
    og_state = game.get_state()
    new_state = None
    counter = 0
    epsiol = .3

    while True:
        try:
            narr = None
            if random.uniform(0,1)>epsiol:
                k,narr =rnn_model.decide(og_state)
            else:
                k = random.randint(0,len(actions)-1)
            new_state, reward, is_loss = game.move(k)

            # print(actions[k],reward)
            # if narr is not None:
                # print("expected: ",list(zip(actions,map(int,narr))))
            memory.append(Memory(og_state,k,new_state,reward,False))
            og_state=new_state
            if is_loss:

                game.restart_loss()

            if counter%30==0:
                game.pause_game()
                print("train...")
                rnn_model.double_train(memory,1,.7,min(len(memory),600))
                game.unpause_game()
            if counter%150==0:
                
                game.pause_game()
                print("saving ...")
                double_model.save("downwellmodel.h5")
                game.unpause_game()
            counter+=1

        except Exception as e:
            traceback.print_exc()
            memory.pop()
            input("Press enter to continue (1 second delay)")
            sleep(1)
            print("starting....")


def play_downwell():
    # shell = win32com.client.Dispatch("WScript.Shell")
    # os.startfile(downwell_file)
    # input("navigate to the point where you are actualy in the game and press enter")
    memory = deque(maxlen=10000)
    sleep(1)
    game = DGame(.001,None)

    actions = ["left","right","jump","jleft","jright"]
    double_model = DoubleDAgent(game.screen_dimensions,len(actions),"downwellmodel.h5")
    og_state = game.get_state()
    new_state = None
    counter = 0
    epsiol = .3

    while True:
        try:
            narr = None
            if random.uniform(0,1)>epsiol:
                k,narr =double_model.decide(og_state)
            else:
                k = random.randint(0,len(actions)-1)
            new_state, reward, is_loss = game.move(k)

            # print(actions[k],reward)
            # if narr is not None:
                # print("expected: ",list(zip(actions,map(int,narr))))
            memory.append(Memory(og_state,k,new_state,reward,False))
            og_state=new_state
            if is_loss:

                game.restart_loss()

            if counter%30==0:
                game.pause_game()
                print("train...")
                double_model.double_train(memory,1,.7,min(len(memory),600))
                double_model.double_train(memory,0,.7,min(len(memory),600))
                game.unpause_game()
            if counter%150==0:
                
                game.pause_game()
                print("saving ...")
                double_model.save("downwellmodel.h5")
                game.unpause_game()
            counter+=1

        except Exception as e:
            traceback.print_exc()
            memory.pop()
            input("Press enter to continue (1 second delay)")
            sleep(1)
            print("starting....")


counter = 0
def gather_human_gameplay():
    game = DGame()
    actions = ["left", "right", "jump", "jleft", "jright"]
    double_model = DoubleDAgent(game.screen_dimensions, len(actions), "downwellmodel_human.h5")
    interval = .01
    og_time = time.time()
    keys_down = set()
    def on_press(key:pynput.keyboard.Key):
        lock = threading.Lock()
        with lock:
            if key not in keys_down:
                keys_down.add(key)
    def on_release(key):
        lock = threading.Lock()
        with lock:
            keys_down.remove(key)
    l= Listener(on_press=on_press,on_release=on_release)
    l.start()
    print("starting")
    lock = threading.Lock()
    prev_state = game.get_state()
    counter= 0
    memories = deque(maxlen=50000)
    while True:
        sleep(interval)
        state = game.get_state()
        if game.is_loss(state):
            print("asleep zzzz")
            time.sleep(5)
            print("awake")
        with lock:
            is_left = Key.left in keys_down
            is_right = Key.right in keys_down
            is_jump = Key.space in keys_down
        action  =-1
        if is_left:
            action = 0
            if is_jump:
                action =3
        elif is_right:
            action = 1
            if is_jump:
                action = 4
        elif is_jump:
            action =2
        if action==-1:
            continue
        reward,is_loss = game.calculate_reward(state)
        print(action,reward)
        memories.append(Memory(prev_state,action,state,reward,False))
        prev_state = state
        if counter%15==0:
            double_model.double_train(memories,1,.7,256)
            double_model.double_train(memories, 0, .7, 256)
        if counter%100==0:
            double_model.save("downwellmodel_human.h5")
            print("saved ..")
        counter+=1
# gather_human_gameplay()


play_downwell()
