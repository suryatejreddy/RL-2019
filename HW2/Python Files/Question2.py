#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


#Create an index map to uniquely identify each cell in the grid with a single number
index_map = [[(i + 5*j) for i in range(5)] for j in range(5)]

#LHS of the bellman equations
lhs = np.zeros((25,25))

#RHS of the bellman equation
rhs = np.zeros((25,1))


# In[3]:


#Function that returns the next state and reward
#The function takes as input the current state and the action performed
def get_next_state_and_reward(row, col, action):
    #The following part of the code handles the special cases where the reward is 10 and 5
    if row == 0:
        if col == 1:
            reward = 10
            state = 21
            return state, reward
        if col == 3:
            reward = 5
            state = 13
            return state, reward
        
    #All other cases are uniform. The action variable denotes up,left,down or right
    x = col + action[0]
    y = row + action[1]
    state = index_map[row][col]
    
    #If the object doesn't fall of the grid, the reward is 0 and state is updated
    if (x > -1 and x < 5) and (y > - 1 and y < 5):
        state = index_map[y][x]
        reward = 0
    else:
        #This is for when the object falls of, the state is the same and reward is -1
        reward = -1
    return state, reward


# In[4]:


allowed = [[-1,0],[1,0],[0,-1],[0,1]]
#Iterate over all cells
for row in range(5):
    for col in range(5):
        cur = index_map[row][col]
        lhs[cur][cur] = 1
        for action in allowed:
            next_, reward = get_next_state_and_reward(row, col, action) 
            lhs[cur][next_] -= 0.25 * 0.9 #0.25 denotes equal probability and 0.9 is the gamma value
            rhs[cur] += 0.25 * reward #0.25 denotes equal probability


# In[5]:


#Use a numpy solver to get the result of the equations.
soln = np.linalg.solve(lhs,rhs)
soln = np.around(soln,1)
soln = soln.reshape((5,5))
soln


# 
