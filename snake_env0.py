import gym
import numpy as np

from gym import spaces
from typing import List, Tuple, Dict, Any, Optional, Union

from stable_baselines3.common.env_checker import check_env

import platform
import os

from collections import deque

__version__ = "0.0.0"

class Snake(gym.Env):
    def __init__(self,
                 grid_size=(8, 8)):

        self.__version__ = __version__
        self.body = [(0, 0)]
        self.direction_vec = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
        self.direction = 0
        self.food = (0, 0)
        self.board = np.zeros(grid_size, dtype=np.uint8)
        self.board_shape = np.array(grid_size)

        self.now = 0
        self.last_eat = 0
        self.max_time = 4 * self.board_shape.sum()


        #self.observation_space =spaces.Box(low=-1, high=100, shape=(grid_size[0]*grid_size[1]+1,) , dtype=int)  # TODO: define observation space
        #self.observation_space =spaces.Box(low=0, high=grid_size[0]+1, shape=(grid_size[0]*grid_size[1]*2+1,) , dtype=int)
        self.observation_space = spaces.Box(low=0, high=self.board_shape[0] * self.board_shape[1] + 1,
                                            shape=(self.board_shape[0] * self.board_shape[1] * 2 + 4, ), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.reset()

    def get_obs(self):
        """
        obs = np.zeros(self.board_shape, dtype=int)
        for i, body in enumerate(self.body, start=1):
            obs[body] = 1 #i 
        body = self.body
        obs[body[0][0]][body[0][1]]=3
        obs[body[len(body)-1][0]][body[len(body)-1][1]]=2
        obs[self.food[0]][self.food[1]]=-1
        return np.concatenate([obs.flatten(),[len(self.body)]] )
        """
        
        """ #내가 좌표로 observation준 방식
        obs = np.zeros(self.board_shape[0]*self.board_shape[1]*2+1,dtype=int)
        cur = 0
        obs[cur]=self.food[0]+1
        cur+=1
        obs[cur]=self.food[1]+1
        cur+=1
        obs[cur]=len(self.body)
        cur+=1
        for i in range(len(self.body)):
            obs[cur]=self.body[i][0]+1
            cur+=1
            obs[cur]=self.body[i][1]+1
            cur+=1

        return obs
        """
        candidates = self.body[0] + self.direction_vec
        indices = np.arange(4)
        bound_condition = np.logical_and((candidates >= 0).all(axis=1), (candidates < self.board.shape).all(axis=1))
        indices = indices[bound_condition]
        candidates = candidates[bound_condition]
        body_condition = self.board[candidates[:, 0], candidates[:, 1]] == 0
        indices = indices[body_condition]
        action_mask = np.zeros(4, dtype=np.uint8)
        action_mask[indices] = 1

        obs = [self.food]
        obs.extend(self.body)
        obs.extend([(-1, -1)] * (self.board.shape[0] * self.board.shape[1] - len(self.body) - 1))
        obs = np.array(obs) + 1
        obs = np.concatenate([obs.flatten(), action_mask])
        return obs
        # TODO: return observation
        

    def reset(self):
        self.board = np.zeros(self.board_shape, dtype=np.uint8)
        #random_init = np.random.randint(2,20)  ############ 학습시 이거
        random_init = 2 ############ play시 이거 -> 근데 학습도 이거로하는게 안정감있는듯..?
        
        self.body = [ (0,0) ]
        
        
        """ 
        self.body = [(np.random.randint(2, self.board_shape[0]-2),
                      np.random.randint(2, self.board_shape[1]-2))]
        
        init_body1 = np.random.choice(4) #0동 1서 2남 3북
        self.body.append( tuple( np.array(self.body[0])  + self.direction_vec[init_body1] ) )
        init_body2 = np.random.choice(4)
        while( (init_body1<2 and init_body2==init_body1+2) or (init_body1>=2 and init_body2==init_body1-2   ) ):
            init_body2 = np.random.choice(4)
        self.body.append( tuple( np.array(self.body[1])  + self.direction_vec[init_body2] ) )
        """
        
        self.body = [(np.random.randint(3, self.board_shape[0]-2),
                      np.random.randint(3, self.board_shape[1]-2))]

        for _ in range(random_init):
            cnt=0
            while True:
                cnt+=1
                vec = self.direction_vec[np.random.choice(4)]
                pos = np.array(self.body[-1]) + vec
                pos2 = np.array(self.body[0]) + vec
                if np.logical_and(pos >= 0, pos < self.board_shape).all() and tuple(pos) not in self.body:
                    self.body.append(tuple(pos))
                    break
                if np.logical_and(pos2 >= 0, pos2 < self.board_shape).all() and tuple(pos2) not in self.body:
                    self.body.insert(0,tuple(pos2))
                    break
                if cnt>=100:
                    break
        
        # TODO: reset body, check board filled correctly
        self.direction = 0

        self.food = self._generate_food()

        self.now = 0
        self.last_eat = 0

        return self.get_obs()

    def render(self, mode="human"):
        if mode == "char":
            black_square = chr(9608) * 2
            white_square = chr(9617) * 2
            # food = chr(9679) * 2
            food = chr(9675) * 2
        else:
            black_square = chr(int('2b1b', 16))
            white_square = chr(int('2b1c', 16))
            #food = chr(int('1f34e', 16))
            #food = chr(int('1f7e7', 16))
            food = '@'
        def encode(v):
            if v == 0:
                return white_square
            elif v > 0:
                return black_square
            elif v == -1:
                return food

        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')

        render_board = self.board.astype(int).copy()
        food_pos = self.food
        render_board[food_pos] = -1
        render_board = np.vectorize(encode)(render_board)
        for row in render_board:
            print(''.join(row))

    def step(self, action):
        self.now += 1
        self.direction = action

        done, info = False, {}
        new_head = self.body[0] + self.direction_vec[self.direction]
        #self.board[self.body.pop()] -= 1
        self.board[self.body.pop()] = 0

        bound_condition = (new_head >= 0).all() and (new_head < self.board_shape).all()
        body_condition = tuple(new_head) not in self.body
        starve_condition = self.now - self.last_eat <= self.max_time
        if bound_condition and body_condition and starve_condition:
            self.body.insert(0, tuple(new_head))
            self.board[tuple(new_head)] = 1
            if self.food == tuple(new_head):
                reward = 100
                self.last_eat = self.now
                self.body.append(self.body[-1])
                #self.board[self.body[-1]] += 1
                self.board[self.body[-1]] = 1
                self.food = self._generate_food()
            else:
                reward = self.heuristic(self.body, self.food, self.direction) #* 0.5
        else:
            done = True
            reward = -100
            if not bound_condition:
                msg = 'out of body'
            elif not body_condition:
                msg = 'body'
            else:
                msg = 'timeout'
            info['mgs'] = msg
        return self.get_obs(), reward, done, info
    
    
    def heuristic(self, body, food, direction): #매번 최단거리로 이동하며, 애초에 구멍(뱀에 의해 board가 쪼개지는것)이 안생기게 한다면?
        # TODO: calculate supplement reward if needed
        """
        head = np.array(self.body[0])
        food = np.array(self.food)
        return -np.linalg.norm(head - food) / self.board_shape.max()
        """
        head = np.array(self.body[0])
        food = np.array(self.food)
        head_before = np.array(self.body[1])
        
        board_copy = self.board.copy()
        for i, _body in enumerate(body, start=1):
            board_copy[_body] = len(body)-i+1 #역순으로 번호붙임
        board_copy[food[0]][food[1]]=0
        queue = deque()
        
        def bfs(start,destination): #통과 가능여부 및 최단거리를 알려줌 #최단거리로 가능여부라 돌아가는길은 못찾음
            #근데 돌아가는길 못찾는거는 밑에 bfs2에서 항상 길이 존재하도록 이동하게 학습하면 문제없을듯?
            distance = np.zeros(self.board_shape) #최단거리를 구하기 위함
            is_visit=np.zeros(self.board_shape) #방문여부
            queue.append(start)
            is_visit[start[0]][start[1]] = 1
            while(queue):
                cur_loc = queue.popleft()
                for i in range(4):
                    new_loc = cur_loc + self.direction_vec[i]
                    if( (new_loc >= 0).all() and (new_loc < self.board_shape).all() ): # 범위 검사
                        if (not is_visit[new_loc[0]][new_loc[1]]): # 방문여부 검사
                            if(board_copy[new_loc[0]][new_loc[1]] <= distance[cur_loc[0]][cur_loc][1]+1 ): # 통과가능 여부 검사
                            #뱀의 인덱스보다 길어야 통과 가능함
                                is_visit[new_loc[0]][new_loc[1]] = 1 #방문표시
                                queue.append(new_loc) #큐에 추가
                                distance[new_loc[0]][new_loc[1]] = distance[cur_loc[0]][cur_loc[1]]+1 #거리 저장
                                if(new_loc[0]==destination[0] and new_loc[1]==destination[1]): #목표 도달시
                                    queue.clear()
                                    return distance[new_loc[0]][new_loc[1]]
            return 0
        
        def bfs2(): #빈 공간이 뱀에 의해 몇개로 쪼개지는지 검사 
            #나중에는 항상 사과를 먹고 나오는길이 존재하는지 검사하도록 개선 가능할 듯
            ans=0
            is_visit=np.zeros(self.board_shape) #방문여부
            for i in range(self.board_shape[0]):
                for j in range(self.board_shape[1]):
                    now = (i,j)
                    if(not is_visit[now[0]][now[1]] and board_copy[now[0]][now[1]]==0): #방문하지 않았고 해당위치에 뱀이 없다면
                        ans+=1
                        is_visit[now[0]][now[1]] = 1 #방문표시
                        queue.append(now) #큐게 추가
                        while(queue): #bfs돌림
                            cur_loc = queue.popleft()
                            for i in range(4):
                                new_loc = cur_loc + self.direction_vec[i]
                                if( (new_loc >= 0).all() and (new_loc < self.board_shape).all() ): # 범위 검사
                                    if (not is_visit[new_loc[0]][new_loc[1]]): # 방문여부 검사
                                        if(board_copy[new_loc[0]][new_loc[1]]==0): #뱀의 위치가 아니라면
                                            is_visit[new_loc[0]][new_loc[1]] = 1 #방문표시
                                            queue.append(new_loc) #큐에 추가
            return ans
        
        is_divided_by_two = bfs2()
        if(is_divided_by_two>1): #2개로 쪼개지면 패널티
            return -10
        
        board_copy[head[0]][head[1]]=0 #머리는 0으로
        value_of_head = bfs(head,food)
        board_copy[head_before[0]][head_before[1]]=0 #이전머리도 0으로
        value_of_head_before = bfs(head_before,food)
        if(value_of_head==0): #만약 최단거리로 가는 경우가 막혀있다면
            return -10 
        if(value_of_head < value_of_head_before ): #이전보다 가까이 갔다면
            return -1 * value_of_head / self.board_shape[0] #-0.125 ~ -5정도 범위
        else:
            return -3 * value_of_head / self.board_shape[0] #-0.375 ~ -15정도 범위
        

    def _generate_food(self):
        while True:
            food = (np.random.randint(0, self.board_shape[0]),
                    np.random.randint(0, self.board_shape[1]))
            if food not in self.body:
                # self._record_food.append(food)
                return food


#if __name__ == "__main__":
#    env = Snake()
#    print(env.body,env.food)
#    env.heuristic(env.body, env.food, env.direction)
