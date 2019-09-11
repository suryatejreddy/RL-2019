#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import optimize


# In[2]:


#Create an index map to uniquely identify each cell in the grid with a single number
index_map = [[(i + 5*j) for i in range(5)] for j in range(5)]

#LHS of the bellman equations
lhs = np.zeros((25 * 4,25))

#RHS of the bellman equation
rhs = np.zeros((25 * 4,1))


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
actions = {0 : "Left", 1 : "Right", 2 : "Up", 3 : "Down" }
#For this part, optimality equation is non linear, 
#because of the max variable present, so we can setup 
#Ax >= b inequality for all the possible actions and then solve

#This gives us 100 inequalities in 25 variables to solve
#Every action has its own reward and next state value 
#THe optimal state value function will be greater than or equal to all these values and hence the inequality holds

#Iterate over all cells
for row in range(5):
    for col in range(5):
        cur = index_map[row][col]
        action_counter = 0
        for action in allowed:
            next_, reward = get_next_state_and_reward(row, col, action) 
            lhs[cur*4 + action_counter][cur] -= 1.0
            lhs[cur*4 + action_counter][next_] += 0.9  #0.9 is the gamma value
            rhs[cur*4 + action_counter] -= reward#0.25 denotes equal probability
            action_counter += 1


# ## V* - Optimal Value Function

# In[5]:


soln = np.asarray(optimize.linprog(np.ones(25), lhs, rhs).x)
soln = np.around(soln,1)
soln = soln.reshape((5,5))
soln


# In[6]:


v_star = soln.reshape(25,1)
pi = np.zeros((25,4))
for row in range(5):
    for col in range(5):
        cur = index_map[row][col]
        pi_cur = pi[cur]
        q_temp = []
        pi_new = np.zeros(4)
        action_counter = 0
        for action in allowed:
            next_, reward = get_next_state_and_reward(row, col, action)
            q_temp.append(v_star[next_][0])
        optim_actions = np.argwhere(q_temp == np.amax(q_temp))
        optim_actions = optim_actions.flatten().tolist()
        num = len(optim_actions)
        pi_new[optim_actions] = 1.0/num
        pi[cur] = pi_new


# ## Optimal Policy

# In[7]:


ans = [["" for i in range(5)] for j in range(5)]
for i in range(5):
    for j in range(5):
        cur = index_map[i][j]
        fav_actions = pi[cur]
        st = ""
        for k in range(4):
            if fav_actions[k] > 0:
                st += actions[k] + " "
        ans[i][j] = st
ans

