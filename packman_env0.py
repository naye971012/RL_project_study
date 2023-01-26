import gym
import numpy as np

from gym import spaces
from typing import List, Tuple, Dict, Any, Optional, Union

from stable_baselines3.common.env_checker import check_env

import platform
import os

from collections import deque

__version__ = "0.0.0"

class packman(gym.Env):
    def __init__(self):
        self.__version__ = __version__ 
        
        map_0 =   [[0,0,0,0,0,0,0,0,0,0,0,0,0], #13*11 array, (6,6) is start of the ghost
            [1,0,1,0,1,1,1,1,1,0,1,0,1],  #9 is coin that packman has to eat
            [0,0,1,0,0,0,1,0,0,0,1,0,0],  #4,5 is ghost that chase packman
            [0,1,1,0,1,0,0,0,1,0,1,1,0],  #7 is packman
            [0,0,0,0,1,1,1,1,1,0,0,0,0],  #1 is wall
            [0,1,1,0,0,0,0,0,0,0,1,1,0],
            [0,0,1,0,1,1,1,1,1,0,1,0,0],
            [1,0,1,0,0,0,1,0,0,0,1,0,1],
            [0,0,0,0,1,0,1,0,1,0,0,0,0],
            [0,1,1,1,1,0,1,0,1,1,1,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0]] 
         
        map_1 =  [[1,1,1,1,1,1,1,1,1,1,1,1,1], #13*11 array, (6,6) is start of the ghost
            [1,0,0,0,1,0,0,0,1,0,0,0,1],  #9 is coin that packman has to eat
            [1,0,1,0,0,0,1,0,0,0,1,0,1],  #4,5 is ghost that chase packman
            [1,0,1,0,1,0,1,0,1,0,1,0,1],  #7 is packman
            [1,0,0,0,1,0,1,0,1,0,0,0,1],  #1 is wall
            [1,0,1,0,0,0,0,0,0,0,1,0,1],
            [1,0,0,0,1,0,1,0,1,0,0,0,1],
            [1,0,1,0,1,0,1,0,1,0,1,0,1],
            [1,0,1,0,0,0,1,0,0,0,1,0,1],
            [1,0,0,0,1,0,0,0,1,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1]]  
        
        self.map_init = np.array( #inital map of packman 팩맨(7)보다 작은 수(벽(1),유령(4,5))는 피하고 코인(9)는 먹도록 의도했습니다.
            map_0
        )
        self.map = self.map_init.copy() #real map that we use
        self.map_shape = np.array( (11,13) )
        self.direction_vec = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
        self.packman = (0,0)
        self.ghost_start=(5,6)
        self.ghost_1 = (0,0) #chase packman by shortest uclidian distance
        self.ghost_2 = (0,0) #goto packman_location+packman_direction*2 by shortest uclidian distance
        self.direction =0
        self.ghost_time = 0 #uses for ghost pattern / repeat every 27 second
        self.cur_time = 0 
        self.eat_num = 0 #how many coins packman ate
        self.max_time = self.map_shape[0]*self.map_shape[1]/2
        
        self.observation_space = spaces.Box(low=0,high=30, shape=(1,self.map_shape[0],self.map_shape[1],) , dtype=int)
        #self.observation_space = spaces.Box(low=0,high=30, shape=( self.map_shape[0]*self.map_shape[1] +2, ) , dtype=int)
        self.action_space = spaces.Discrete(4)
        self.reset()
    
    def get_obs(self): #처음에는 snake_game과 같이 좌표로 obs를 주었지만, observation_space가 너무 커져 map을 flatten하는 방식으로 주었습니다.
    
        obs = [self.ghost_time]
        obs.extend([self.eat_num])
        map_copy = self.map.copy()
        if self.packman[0]>=0 and self.packman[1]>=0 and self.packman[0]<self.map_shape[0] and self.packman[1]<self.map_shape[1]:
            map_copy[self.packman[0]][self.packman[1]]=7
        if self.ghost_1[0]>=0 and self.ghost_1[1]>=0 and self.ghost_1[0]<self.map_shape[0] and self.ghost_1[1]<self.map_shape[1]:
            map_copy[self.ghost_1[0]][self.ghost_1[1]]=4
        if self.ghost_2[0]>=0 and self.ghost_2[1]>=0 and self.ghost_2[0]<self.map_shape[0] and self.ghost_2[1]<self.map_shape[1]:
            map_copy[self.ghost_2[0]][self.ghost_2[1]]=4
        
        return [map_copy]
        
        for i in map_copy.flatten():
            obs.extend([i])
        #print(obs)
        
        return obs
    """    
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                if(self.map[i][j]==1):
                    obs.append(i)
                    obs.append(j)
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                if(self.map[i][j]==0):
                    obs.append(0)
                    obs.append(0)
                elif(self.map[i][j]==2):
                    obs.append(i)
                    obs.append(j)
        return obs
    """
                    
    def reset(self) -> Any:
        self.map = self.map_init.copy()
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                if(self.map[i][j]==0):
                    self.map[i][j]=9 #reset coin
        
        self.ghost_1 = self.ghost_start #reset ghost location
        self.ghost_2 = self.ghost_start #reset ghost location
        
        #random packman_start location
        self.packman = (0,0)
        
        x = np.random.randint(0,self.map_shape[0])
        y = np.random.randint(0,self.map_shape[1])
        while (self.map[x][y]!=1):
            x = np.random.randint(0,self.map_shape[0])
            y = np.random.randint(0,self.map_shape[1])
        self.packman = (x,y)
        
        self.cur_time=0
        self.eat_num=0
        self.ghost_time=0
        
        return self.get_obs()
    
    def ghost_move(self): #실제 팩맨에서 유령은20초 추적 7초 휴식의 단계를 가집니다. 20초동안은 팩맨과 가까워지고,
        #7초 동안은 팩맨과 멀어지도록 설계하였습니다.
        #ghost_1은 팩맨의 지점을 목표로, ghost_2는 팩맨이 바라보는지점+2를 목표로 유클리드 거리를 계산해 이동합니다
        
        going=0
        #chase for 20 second, scatter for 7 second. repeat it
        if(self.ghost_time<20):
            x=100
            minus=1
        else:
            x=0
            minus=-1
        
        min=x
        for i in range(4): #ghost_1 is moving by eculidian distance
            next_loc = self.ghost_1 + self.direction_vec[i]
            if next_loc[0]>=0 and next_loc[1]>=0 and next_loc[0]<self.map_shape[0] and next_loc[1]<self.map_shape[1]:
                if(self.map[next_loc[0]][next_loc[1]]!=1):
                    next_distance = np.linalg.norm(next_loc-self.packman)
                    if(next_distance * minus < min * minus ):
                        going=i
                        min=next_distance
        self.ghost_1 = self.ghost_1 + self.direction_vec[going]
        
        if self.eat_num<10: #ghost_2 is only move when coin>=10
            return

        min=x
        for i in range(4): #ghost_2 is moving by 팩맨보고있는방향+2를 위치로
            next_loc = self.ghost_2 + self.direction_vec[i]
            if next_loc[0]>=0 and next_loc[1]>=0 and next_loc[0]<self.map_shape[0] and next_loc[1]<self.map_shape[1]:
                if(self.map[next_loc[0]][next_loc[1]]!=1):
                    next_distance = np.linalg.norm(next_loc-(self.packman + 2 * self.direction_vec[self.direction] ))
                    
                    if(next_distance * minus < min * minus ):
                        check = self.ghost_2 + self.direction_vec[i]
                        if(check[0]>=0 and check[1]>=0 and check[0]<self.map_shape[0] and check[1]<self.map_shape[1]):
                            going=i
                            min=next_distance
        self.ghost_2 = self.ghost_2 + self.direction_vec[going]
        
        
    def step(self, action): #snake_game의 step과 비슷하며,
        
        self.ghost_move() # move ghost by predetermined pattern
        
        done, info = False, {}
        self.direction = action
        self.packman = self.packman + self.direction_vec[action] #move packman
        self.cur_time+=1
        self.ghost_time = (self.ghost_time+1)%27
        
        bound_condition = (self.packman >= 0).all() and (self.packman < self.map_shape).all()
        starve_condition = self.cur_time <= self.max_time
        ghost_condition = (self.packman[0]==self.ghost_1[0] and self.packman[1]==self.ghost_1[1]) or (self.packman[0]==self.ghost_2[0] and self.packman[1]==self.ghost_2[1])
        
        if bound_condition and starve_condition and not ghost_condition:
            if self.map[self.packman[0]][self.packman[1]]==9: # if packman eat coin
                self.cur_time=0
                self.eat_num+=1
                self.map[self.packman[0]][self.packman[1]]=0
                reward = 50 + self.heuristic()
            elif self.map[self.packman[0]][self.packman[1]]==1: # if packman goto wall
                done = True
                reward = -1000
                info['msf']='wall'
            else:
                reward = self.heuristic()
        
        else:
            done = True
            reward = -1000
            if not bound_condition: #out of map
                msg = 'out of map'
            elif ghost_condition: #packman meet ghost
                msg = 'ghost'
                reward=-500
            else:
                msg = 'timeout' #packman is looping
            info['mgs'] = msg
        
        return self.get_obs(), reward, done, info

    def heuristic(self): #코인을 많이 먹는 것이 중요하므로, 코인을 먹지 못한 시간만큼 리워드가 점차 감소합니다.
        #유령과 가까울수록, 리워드가 감소합니다 
        # 
        # -수정-> 코인과 가까울수록 리워드가 증가합니다.
        # -수정-> 코인과 가까울수록, 유령과 멀수록 리워드가 증가합니다.
        queue = deque()
        
        def bfs(start,destination): #최단거리를 알려줌 #최단거리로 가능여부라 돌아가는길은 못찾음
            distance = np.zeros(self.map_shape) #최단거리를 구하기 위함
            is_visit=np.zeros(self.map_shape) #방문여부
            queue.append(start)
            is_visit[start[0]][start[1]] = 1
            while(queue):
                cur_loc = queue.popleft()
                for i in range(4):
                    new_loc = cur_loc + self.direction_vec[i]
                    if( (new_loc >= 0).all() and (new_loc < self.map_shape).all() ): # 범위 검사
                        if (not is_visit[new_loc[0]][new_loc[1]] and self.map[new_loc[0]][new_loc[1]]!=1): # 방문여부 검사 + 벽 여부 검사
                                is_visit[new_loc[0]][new_loc[1]] = 1 #방문표시
                                queue.append(new_loc) #큐에 추가
                                distance[new_loc[0]][new_loc[1]] = distance[cur_loc[0]][cur_loc[1]]+1 #거리 저장
                                if(new_loc[0]==destination[0] and new_loc[1]==destination[1]): #목표 도달시
                                    queue.clear()
                                    return distance[new_loc[0]][new_loc[1]]
        
        def bfs2(start):
            distance = np.zeros(self.map_shape) #최단거리를 구하기 위함
            is_visit=np.zeros(self.map_shape) #방문여부
            queue.append(start)
            is_visit[start[0]][start[1]] = 1
            while(queue):
                cur_loc = queue.popleft()
                for i in range(4):
                    new_loc = cur_loc + self.direction_vec[i]
                    if( (new_loc >= 0).all() and (new_loc < self.map_shape).all() ): # 범위 검사
                        if (not is_visit[new_loc[0]][new_loc[1]] and self.map[new_loc[0]][new_loc[1]]!=1): # 방문여부 검사 + 벽 여부 검사
                                is_visit[new_loc[0]][new_loc[1]] = 1 #방문표시
                                queue.append(new_loc) #큐에 추가
                                distance[new_loc[0]][new_loc[1]] = distance[cur_loc[0]][cur_loc[1]]+1 #거리 저장
                                if(self.map[new_loc[0]][new_loc[1]]==9): #코인에 도달하면
                                    queue.clear()
                                    return distance[new_loc[0]][new_loc[1]]
        ghost_1_distance = bfs(self.packman,self.ghost_1)
        ghost_2_distance = bfs(self.packman,self.ghost_2)
        if ghost_1_distance==None or ghost_2_distance==None:
            distance_heuristic=0
        else:
            distance_heuristic = min(ghost_1_distance,ghost_2_distance)
        
        #nearest_coin = bfs2(self.packman)
        #if nearest_coin==None:
        #    nearest_coin=0
        
        #return -np.sqrt(self.cur_time) + np.sqrt(distance_heuristic)
        #return -np.sqrt(self.cur_time) - np.sqrt(nearest_coin)
        #return -np.sqrt(nearest_coin) + np.sqrt(distance_heuristic) 
        return np.sqrt(distance_heuristic) 
    
    def render(self): #미완성
        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')


        map_copy = self.map.copy()
        if self.packman[0]>=0 and self.packman[1]>=0 and self.packman[0]<self.map_shape[0] and self.packman[1]<self.map_shape[1]:
            map_copy[self.packman[0]][self.packman[1]]=7
        
        map_copy[self.ghost_1[0]][self.ghost_1[1]]=4
        map_copy[self.ghost_2[0]][self.ghost_2[1]]=5 
        
        cnt=0
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                x = map_copy[i][j]

                if x==1: #wall
                    print(chr(int('2b1c', 16)),end='')
                elif x==0: #empty
                    print('  ',end='')
                elif x==7: #packman
                    print('◀▶',end='')
                elif x==9: #coin
                    print(chr(int('2b1b', 16)),end='')
                else: #ghost
                    print('◁▷',end='')
            print('')
        #print(map_copy) ♡ ♥
    

if __name__ == "__main__":
    env = packman()
#    print(env.packman)
#    x = env.step(1)
#    print(x)
#"""