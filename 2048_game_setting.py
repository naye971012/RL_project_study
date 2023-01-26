import gym
import numpy as np

from gym import spaces
from typing import List, Tuple, Dict, Any, Optional, Union

from stable_baselines3.common.env_checker import check_env

import platform
import os
import time
from collections import deque

class game_2048(gym.Env):
    def __init__(self):

        self.action_space=spaces.Discrete(4)
        self.observation_space=spaces.Box(low=0,high=1024,shape=(20,),dtype=int)
        
        self.map_size=4
        self.map_init = []
        for i in range(self.map_size):
                self.map_init.append([])
                for j in range(self.map_size):
                        self.map_init[i].append(0)
        self.map = self.map_init
        self.score=0
        
        self.reset()
    
    def make_new_number(self,cnt):
        
        done = True
        for i in range(self.map_size):
            for j in range(self.map_size):
                if(self.map[i][j]==0):
                    done=False
        if done:
            return 0
        
        for _ in range(cnt):
                while(1):
                        a= np.random.randint(0,self.map_size)
                        b= np.random.randint(0,self.map_size)
                        if(self.map[a][b]==0):
                                self.map[a][b]=2
                                break    
        return 1
    def reset(self) -> Any:
        
        for i in range(self.map_size):
            for j in range(self.map_size):
                self.map[i][j]=0
        self.score = 0
        
        self.make_new_number(2)
        
        return self.get_obs()
    
    def get_obs(self):
            #obs = np.array(self.map)
            obs=[]
            for i in range(self.map_size):
                    for j in range(self.map_size):
                        obs.append(np.log2(self.map[i][j]+1))
        
            action_space = [0,0,0,0]
            for i in range(self.map_size-1):
                for j in range(self.map_size-1):
                        x=self.map[i][j]
                        y=self.map[i+1][j]
                        z=self.map[i][j+1]
                        if(x==y):
                                action_space[0]=1
                                action_space[1]=1
                        if(x==z):
                                action_space[2]=1
                                action_space[3]=1
                            
            for i in action_space:
                    obs.append(i)

            return obs
    
    def move_map(self,is_move,direction): #move일 경우 map에서 움직임. check일 경우 map_copy에서 움직임(map에 영향을 주지 않음)
        
        if is_move=='move':
                _map = self.map.copy()
                
                if direction=='up':
                        for i in range(self.map_size):
                                queue = deque()
                                for j in range(self.map_size):
                                        if _map[j][i]:
                                                queue.append(_map[j][i])
                                                _map[j][i]=0
                                idx = 0
                                while queue:
                                        if len(queue) > 1:
                                                a,b = queue.popleft(),queue.popleft()
                                                if a==b:
                                                        _map[idx][i]=a+b
                                                else:
                                                        _map[idx][i]=a
                                                        queue.appendleft(b)
                                                idx+=1
                                        else:
                                                _map[idx][i] = queue.popleft()

                elif direction=='down':
                        for i in range(self.map_size):
                                queue = deque()
                                for j in range(self.map_size-1,-1,-1):
                                        if _map[j][i]:
                                                queue.append(_map[j][i])
                                                _map[j][i]=0
                                idx = self.map_size-1
                                while queue:
                                        if len(queue) > 1:
                                                a,b = queue.popleft(),queue.popleft()
                                                if a==b:
                                                        _map[idx][i]=a+b
                                                else:
                                                        _map[idx][i]=a
                                                        queue.appendleft(b)
                                                idx-=1
                                        else:
                                                _map[idx][i] = queue.popleft()
                
                elif direction=='left':
                        for i in range(self.map_size):
                                queue = deque()
                                for j in range(self.map_size):
                                        if _map[i][j]:
                                                queue.append(_map[i][j])
                                                _map[i][j]=0
                                idx = 0
                                while queue:
                                        if len(queue) > 1:
                                                a,b = queue.popleft(),queue.popleft()
                                                if a==b:
                                                        _map[i][idx]=a+b
                                                else:
                                                        _map[i][idx]=a
                                                        queue.appendleft(b)
                                                idx+=1
                                        else:
                                                _map[i][idx] = queue.popleft()

                elif direction=='right':
                        for i in range(self.map_size):
                                queue = deque()
                                for j in range(self.map_size-1,-1,-1):
                                        if _map[i][j]:
                                                queue.append(_map[i][j])
                                                _map[i][j]=0
                                idx = self.map_size-1
                                while queue:
                                        if len(queue) > 1:
                                                a,b = queue.popleft(),queue.popleft()
                                                if a==b:
                                                        _map[i][idx]=a+b
                                                else:
                                                        _map[i][idx]=a
                                                        queue.appendleft(b)
                                                idx-=1
                                        else:
                                                _map[i][idx] = queue.popleft()
                
                else:
                        print("wrong input")
                self.map = _map.copy()
    def step(self,action):
        
        #print(self.score)
        
        info = {}
        move_dir = {
                0 : 'up',
                1 : 'down',
                2 : 'left',
                3 : 'right'
        }
        
        self.move_map('move', move_dir[action])
        k = self.make_new_number(1)
        done,reward,info['msg'] = True,-100,'die' #끝난상황 가정
        if k==0: #생성 안되면 죽음
            return self.get_obs(), reward, done, info
        
        for i in range(self.map_size-1):
            for j in range(self.map_size-1):
                x=self.map[i][j]
                y=self.map[i+1][j]
                z=self.map[i][j+1]
                if(x==y or x==z):
                    done=False
                    reward=0
                    info={}
                    break
                if(x==0 or y==0 or z==0):
                    done=False
                    reward=0
                    info={}
                    break
        
        #print(self.map_after,done)
        #time.sleep(0.5)
        
        self.score=0
        for i in range(self.map_size):
            for j in range(self.map_size):
                x=self.map[i][j]
                self.score+=x
                if x==2048 * (2**(self.map_size-4)) : #완성됬다면
                        done=True
                        reward=2048
                        info['msg']='clear'
        if not done:
                reward=reward + self.heuristic()
        return self.get_obs(), reward, done, info
    
    def heuristic(self): #
        
        #return self.coin
        """cnt=0
        for i in range(self.map_size):
                for j in range(self.map_size):
                        if self.map[i][j]==0:
                                cnt+=1
            
        return cnt + np.sqrt( np.argmax(self.map) )""" 
        #최댓값
        #같은 행렬에 같은 숫자->망함 #reward만 return하면 max값들을 안 합치려고 함
        
        _max=np.argmax((self.map))
        reward=0
        x=[]
        cur=2
        for i in range(self.map_size):
                        for j in range(self.map_size):
                                x.append(np.log2(self.map[i][j]+1))
        while(cur<=_max):
                cnt=0
                itr=2
                for i in range(len(x)):
                        if x[i]==cur:
                                cnt+=1
                if(cnt>1):
                        if(itr<1):
                                reward-=(cur)
                        itr-=1
                else:
                        reward+=(cur)
                cur*=2
                
        return reward
        
    def render(self):
        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')
        
        for i in range(self.map_size):
                for j in range(self.map_size):
                        print(self.map[i][j],end=' ')
                print()
        #time.sleep(1)
        
        
"""if __name__ == "__main__":
        env = game_2048()
        
        env.render()
        x=int(input())
        while(x!=5):
                env.step(x)
                env.render()
                x=int(input())"""
