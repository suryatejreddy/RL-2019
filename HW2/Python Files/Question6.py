#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


#Function that returns the next state and reward
#The function takes as input the current state and the action performed
def get_next_state_and_reward(row, col, action):       
    #All other cases are uniform. The action variable denotes up,left,down or right
    x = col + action[0]
    y = row + action[1]
    state = index_map[row][col]
    
    #If the object doesn't fall of the grid, the reward is 0 and state is updated
    if (x > -1 and x < 4) and (y > - 1 and y < 4):
        state = index_map[y][x]
    reward = -1
    return state, reward

def are_similar(a,b):
    for i in range(len(a)):
        if a[i] == 0 and b[i] == 0:
            continue
        elif (a[i] == 0):
            return False
        elif (b[i] == 0):
            return False
        pass
    return True


# ## Policy Iteration

# In[3]:


#Create an index map to uniquely identify each cell in the grid with a single number
index_map = [[(i + 4*j) for i in range(4)] for j in range(5)]

#Value function for all the states
value_states = [0 for i in range(16)]

#Possible Actions
allowed = [[-1,0],[1,0],[0,-1],[0,1]]
actions = {0 : "Left", 1 : "Right", 2 : "Up", 3 : "Down" }

#The limit when we want our algorithm to break
theta = 0.0001

#Assuming a stochastic problem
pi = np.zeros((16,4))
pi.fill(0.25)


# ### To fix the bug mentioned in the question, i took a stochastic setting and created an array to store p(a|s) for each of the 4 action

# In[4]:


while True:
    while True:
        delta = 0
        for row in range(4):
            for col in range(4):
                #Get the state s
                cur = index_map[row][col]
                
                #Skip the terminal states
                if cur == 0 or cur == 15:
                    continue
                
                #Get current value of V(s)
                v_cur = value_states[cur]
                v_new = 0
                action_counter = 0
                
                #Iterate over all actions
                
                for action in allowed:
                    #Get the next state and corresponding reward (s', r)
                    next_, reward = get_next_state_and_reward(row, col, action) 
                    
                    #For each action sum over p(s|a) * 1.0 * (r + v(s'))
                    v_new += pi[cur][action_counter] * 1.0 * (reward + value_states[next_]) 
                    
                    action_counter += 1
                print ("Updated Value of State " + str(cur) + " From " + str(v_cur) + " To " + str(v_new))
                
                #Update the value of V(s)
                value_states[cur] = v_new
                
                #Check the level of update
                delta = max(delta, abs(v_cur - v_new))
        if delta < theta:
            break
    
    policy_stable = True
    for row in range(4):
        for col in range(4):
            cur = index_map[row][col]
            if cur == 0 or cur == 15:
                continue
            #Get the current probabilities for each action
            pi_cur = pi[cur].copy()
            q_temp = []
            
            #Setup new probabilities
            pi_new = np.zeros(4)
            
            for action in allowed:
                next_, reward = get_next_state_and_reward(row, col, action)
                #Store new values of q(s,a)
                q_temp.append(reward + value_states[next_])
                
            #BUG FIX - Get all the actions which have same maximum q
            optim_actions = np.argwhere(q_temp == np.amax(q_temp))
            optim_actions = optim_actions.flatten().tolist()
            
            #Number of optimal actions
            num = len(optim_actions)
            
            #p(a|s) = 1/num optimal
            pi_new[optim_actions] = 1.0/num
            
            #Update this value for all actions
            pi[cur] = pi_new
            if not are_similar(pi_cur, pi_new):
                policy_stable = False
    if (policy_stable):
        break


# In[5]:


value_states[:]


# In[6]:


ans = [["" for i in range(4)] for j in range(4)]
for i in range(4):
    for j in range(4):
        cur = index_map[i][j]
        fav_actions = pi[cur]
        st = ""
        for k in range(4):
            if fav_actions[k] > 0:
                st += actions[k] + " "
        ans[i][j] = st
ans


# ## Value Iteration

# In[7]:


#Create an index map to uniquely identify each cell in the grid with a single number
index_map = [[(i + 4*j) for i in range(4)] for j in range(5)]

#Value function for all the states
value_states = [0 for i in range(16)]

#Possible Actions
allowed = [[-1,0],[1,0],[0,-1],[0,1]]

#The limit when we want our algorithm to break
theta = 0.0001

pi = np.zeros((16,4))
pi.fill(0.25)


# In[8]:


#The code here is same as above, except we use a max function to get the maximum q
while True:
    delta = 0
    for row in range(4):
        for col in range(4):
            cur = index_map[row][col]
            if cur == 0 or cur == 15:
                continue
            v_cur = value_states[cur]
            q_temp = []
            for action in allowed:
                next_, reward = get_next_state_and_reward(row, col, action) 
                q_temp.append(reward + value_states[next_])
            #CHANGE
            v_new = max(q_temp)
            print ("Updated Value of State " + str(cur) + " From " + str(v_cur) + " To " + str(v_new))
            value_states[cur] = v_new
            delta = max(delta, abs(v_cur - v_new))
    if delta < theta:
        break
for row in range(4):
    for col in range(4):
        cur = index_map[row][col]
        if cur == 0 or cur == 15:
            continue
        pi_cur = pi[cur].copy()
        q_temp = []
        pi_new = np.zeros(4)
        action_counter = 0
        for action in allowed:
            next_, reward = get_next_state_and_reward(row, col, action)
            q_temp.append(reward + value_states[next_])
        optim_actions = np.argwhere(q_temp == np.amax(q_temp))
        optim_actions = optim_actions.flatten().tolist()
        num = len(optim_actions)
        pi_new[optim_actions] = 1.0/num
        pi[cur] = pi_new


# In[9]:


value_states


# In[10]:


## Just printing actions that have non zero probability
ans = [["" for i in range(4)] for j in range(4)]
for i in range(4):
    for j in range(4):
        cur = index_map[i][j]
        fav_actions = pi[cur]
        st = ""
        for k in range(4):
            if fav_actions[k] > 0:
                st += actions[k] + " "
        ans[i][j] = st
ans


# ### We obtain the same results using both the methods
