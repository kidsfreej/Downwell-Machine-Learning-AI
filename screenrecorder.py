import time
from typing import Tuple
import cv2
import numpy as np
from PIL import Image
from time import sleep
import pygetwindow as gw
from mss import mss
import cv2
import pyautogui
from game import State
class DState(State):
    def __init__(self,screen):
        self.screen = cv2.cvtColor(screen,cv2.COLOR_BGRA2BGR)
        self.gray = cv2.cvtColor(screen,cv2.COLOR_BGRA2GRAY)
        self.feed =cv2.pyrDown(self.gray)
class DGame:
    def __init__(self,interval=.2,shell=None):
        self.is_paused = False
        self.window = gw.getWindowsWithTitle("Downwell")[0]
        self.shell = shell
        self.screen_dimensions =  (213,120)
        self.score = 0
        self.do_pause = False
        self.interval = interval
    def get_score(self,state:DState):
        left_x = 100
        bottom_y = 55
        num_arr =[]
        for num, template in number_imgs.items():
            res = cv2.matchTemplate(state.gray[:bottom_y,left_x:], template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)

            for pt in loc[1]:
                num_arr.append((pt, num))

        num_arr.sort(key=lambda x: x[0])
        if len(num_arr)>0:
            return int(''.join(map(lambda x:str(x[1]),num_arr)))
        raise Exception("failed to get score")
    def pause_game(self):
        if self.is_paused:
            raise Exception("attempted to pause when alreadp aused")
        self.is_paused=True
        pyautogui.press("esc")
    def unpause_game(self):
        if not self.is_paused:
            raise Exception("attempted to unpause when not paused")
        self.is_paused=False
        pyautogui.press("esc")
    def snap_photo(self):
        win = self.window
        bounding_box = {'top': win.top+31, 'left': win.left+8, 'width': win.width-16, 'height': win.height-9-31}
        with mss() as sct:
            r = np.array(sct.grab(bounding_box))
            return r
    def get_state(self):
        return DState(self.snap_photo())
    def get_hp(self,state):
        threshold = .8
        right_x = 100
        bottom_y = 55
        num_arr = []
        for num, template in number_imgs.items():
            res = cv2.matchTemplate(state.gray[:bottom_y, :right_x], template, cv2.TM_SQDIFF)
            threshold = 1
            loc = np.where(res >= threshold)

            for pt in loc[1]:
                num_arr.append((pt, num))

        num_arr.sort(key=lambda x: x[0])
        print(num_arr)
        if len(num_arr) > 0:
            s = ""
            for num in num_arr:
                if num[1]=="slash":
                    break
                s+=str(num[1])
            return s
        raise Exception("failed to get score")

    def press_direction(self,action):
        if action<3:
            pyautogui.keyDown(["left","right","space"][action])
            return
        pyautogui.keyDown("space")
        pyautogui.keyDown(["left","right"][action-3])
    def unpress_directon(self,action):
        if action < 3:
            pyautogui.keyUp(["left", "right", "space"][action])
            return
        pyautogui.keyUp("space")
        pyautogui.keyUp(["left", "right"][action - 3])
    def is_loss(self,state:DState):
        right_x = 70
        bottom_y = 52
        threshold = .8
        result= cv2.matchTemplate(state.gray[:bottom_y,:right_x],game_over_img,cv2.TM_CCOEFF_NORMED)
        loc = np.where(result>=threshold)
        if len(loc[0])>0:
            return True
        return False
    def calculate_reward(self,state:DState)-> Tuple[int,bool]:
        prev_score = self.score
        self.score = self.get_score(state)
        reward = self.score-prev_score
        reward*=3
        if reward<0:
            reward =50
        is_loss = self.is_loss(state)

        if is_loss:
            reward=-1000
            self.score = 0
        if reward==0:
            reward = -.5
        return reward,is_loss
    def move(self,action)->Tuple[DState,int,bool]:
        if self.do_pause:
            self.unpause_game()
        self.press_direction(action)
        sleep(self.interval)
        self.unpress_directon(action)
        state = self.get_state()
        if self.do_pause:
            self.pause_game()
        reward,is_loss = self.calculate_reward(state)
        return state,reward,is_loss
    def restart_loss(self):
        for i in range(10):
            pyautogui.press("space")
            sleep(0.3)
        sleep(2)
number_img_names = ["font/0.png","font/1.png","font/2.png","font/3.png","font/4.png","font/5.png","font/6.png","font/7.png","font/8.png","font/9.png"]
number_imgs = {i:cv2.imread(v,0) for i,v in enumerate(number_img_names)}
number_imgs["slash"] = cv2.imread("font/slash.png",0)
game_over_img = cv2.imread("font/game_over.png",0)
