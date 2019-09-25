#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import random
RIGHT = 1
LEFT = -1


# In[2]:


def make_move(cur_state, cur_action):
    if cur_state == 5 and cur_action == RIGHT:
        return 6, 1
    else:
        return cur_state + cur_action , 0

def take_action():
    p = random.randint(0,1)
    if p == 0:
        return LEFT
    return RIGHT

def one_td0(episodes, alpha, rmse_flag = False):
    #Init State Values to be 0.5 initially
    values = np.ones(7) * 0.5
    values[0] = 0
    values[-1] = 0

    #Init End States
    end_states = [0,6]
    
    #Plot Values
    break_points = [1,10,100]
    value_plot_ys = []
    value_plot_ys.append(values.copy())
    
    if rmse_flag:
        rmse = np.zeros(episodes)
        targets = [0,1/6,2/6,3/6,4/6,5/6,0]
    
    for episode in range(episodes):
        cur_state = 3
        while cur_state not in end_states:
            cur_action = take_action()
            next_state, cur_reward = make_move(cur_state, cur_action)
            update = cur_reward + values[next_state] - values[cur_state]
            values[cur_state] += alpha * update
            cur_state = next_state 
        if episode in break_points:
            value_plot_ys.append(values.copy())
        if rmse_flag:
            rmse[episode] = np.sqrt(np.mean((values-targets)**2))
    
    if rmse_flag:
        return rmse
    return value_plot_ys

def one_mc(episodes, alpha):
    #Init State Values to be 0.5 initially
    values = np.ones(7) * 0.5
    values[0] = 0
    values[-1] = 0
    
    #Init End States
    end_states = [0,6]
    
    rmse = np.zeros(episodes)
    targets = [0,1/6,2/6,3/6,4/6,5/6,0]
    
    for episode in range(episodes):
        cur_state = 3
        simulations = []
        while cur_state not in end_states:
            cur_action = take_action()
            next_state, cur_reward = make_move(cur_state, cur_action)
            simulations.append([cur_state, cur_reward])
            cur_state = next_state
        g = 0
        reverse = simulations[::-1]
        for sim in reverse:
            cur_state = sim[0]
            cur_reward = sim[-1]
            g += cur_reward
            update = g - values[cur_state]
            values[cur_state] += alpha * update
    
        rmse[episode] = np.sqrt(np.mean((values-targets)**2))
    return rmse
def play_mc(episodes, alpha):
    rmse = np.zeros((100, episodes))
    for i in range(100):
        cur_rmse = one_mc(episodes, alpha)
        rmse[i] = cur_rmse
    return rmse.mean(axis = 0)

def play_td(episodes, alpha):
    rmse = np.zeros((100, episodes))
    for i in range(100):
        cur_rmse = one_td0(episodes, alpha, True)
        rmse[i] = cur_rmse
    return rmse.mean(axis = 0)


# ## Estimated Value

# In[3]:


all_episodes = one_td0(101, 0.1, False)
labels = [0,1,10,100]
for ep in range(len(all_episodes)):
    plt.plot(all_episodes[ep][1:6], label = labels[ep])
plt.plot([1/6,2/6,3/6,4/6,5/6], label = 'True Values')
plt.xlabel('State')
plt.ylabel('Estimated Value')
plt.legend()


# ## RMSE PLOTS

# In[4]:


rmse_mc0 = play_mc(101,0.01)
rmse_mc1 = play_mc(101,0.02)
rmse_mc2 = play_mc(101,0.03)
rmse_mc3 = play_mc(101,0.04)
plt.plot(rmse_mc0, label = 'MC alpha = 0.01')
plt.plot(rmse_mc1, label = 'MC alpha = 0.02')
plt.plot(rmse_mc2, label = 'MC alpha = 0.03')
plt.plot(rmse_mc3, label = 'MC alpha = 0.04')


rmse_td0 = play_td(101,0.05)
rmse_td1 = play_td(101,0.1)
rmse_td2 = play_td(101,0.15)
plt.plot(rmse_td0, label = 'TD alpha = 0.05')
plt.plot(rmse_td1, label = 'TD alpha = 0.1')
plt.plot(rmse_td2, label = 'TD alpha = 0.15')
plt.xlabel('Episode')
plt.ylabel('RMS Error')
plt.legend()

