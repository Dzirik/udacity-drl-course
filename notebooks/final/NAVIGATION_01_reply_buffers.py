# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Template for Final Notebook
# *Version:* `1.0` *(Jupytext, time measurements, logger)*

# <a name="ToC"></a>
# # Table of Content
#
# - [Notebook Description](#0)
# - [General Settings](#1)
#     - [Paths](#1-1)
#     - [Notebook Functionality and Appearance](#1-2)
#     - [External Libraries](#1-3)
#     - [Internal Code](#1-4)
#     - [Constants](#1-5)   
# - [Analysis](#2)   
#     - [Replay Buffer](#2-1)   
#     - [Experience Replay Buffer](#2-2)   
#         - [Simple Examples](#2-2-1)
#         - [Examples with Update of Priorities](#2-2-2)
# - [Final Timestamp](#3)  

# <a name="0"></a>
# # Notebook Description
# [ToC](#ToC) 

# > *Please put your comments about the notebook functionality here.*  

# <a name="1"></a>
# # GENERAL SETTINGS
# [ToC](#ToC)  
# General settings for the notebook (paths, python libraries, own code, notebook constants). 
#
# > *NOTE: All imports and constants for the notebook settings shoud be here. Nothing should be imported in the analysis section.*

# <a name="1-1"></a>
# ### Paths
# [ToC](#ToC)  
#
# Adding paths that are necessary to import code from within the repository.

import sys
import os
sys.path+=[os.path.join(os.getcwd(), ".."), os.path.join(os.getcwd(), "../..")] # one and two up

# <a name="1-2"></a>
# ### Notebook Functionality and Appearance
# [ToC](#ToC)  
# Necessary libraries for notebook functionality:
# - A button for hiding/showing the code. By default it is deactivated and can be activated by setting CREATE_BUTTON constant to True. 
# > **NOTE: This way, using the function, the button works only in active notebook. If the functionality needs to be preserved in html export, then the code has to be incluced directly into notebook.**
# - Set notebook width to 100%.
# - Notebook data frame setting for better visibility.
# - Initial timestamp setting and logging the start of the execution.

from src.utils.notebook_support_functions import create_button, get_notebook_name
from src.utils.logger import Logger
from src.utils.envs import Envs
from pandas import options
from IPython.display import display, HTML

# > Constants for overall behaviour.

LOGGER_CONFIG_NAME = "logger_file_console" # default
PYTHON_CONFIG_NAME = "python_repo" # default
CREATE_BUTTON = False
ADDAPT_WIDTH = False
NOTEBOOK_NAME = get_notebook_name()

options.display.max_rows = 500
options.display.max_columns = 500
envs = Envs()
envs.set_logger(LOGGER_CONFIG_NAME)
envs.set_config(PYTHON_CONFIG_NAME)
Logger().start_timer(f"NOTEBOOK; Notebook name: {NOTEBOOK_NAME}")
if CREATE_BUTTON:
    create_button()
if ADDAPT_WIDTH:
    display(HTML("<style>.container { width:100% !important; }</style>")) # notebook width

# <a name="1-3"></a>
# ### External Libraries
# [ToC](#ToC)  

from numpy import array

# <a name="1-4"></a>
# ### Internal Code
# [ToC](#ToC)  
# Code, libraries, classes, functions from within the repository.

from src.data.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

# <a name="1-5"></a>
# ### Constants
# [ToC](#ToC)  
# Constants for the notebook.
#
# > *NOTE: Please use all letters upper.*

# #### General Constants
# [ToC](#ToC)  

# from src.constants.global_constants import *  # Remember to import only the constants in use
N_ROWS_TO_DISPLAY = 2
FIGURE_SIZE_SETTING = {"autosize": False, "width": 2200, "height": 750}

# #### Constants for Setting Automatic Run
# [ToC](#ToC)  



# #### Notebook Specific Constants
# [ToC](#ToC)  



# <a name="2"></a>
# # ANALYSIS
# [ToC](#ToC)  

# <a name="2-1"></a>
# ## Replay Buffer
# [ToC](#ToC)  
#
# Three different types of actions are tested here.


# +
state_dim = 2
actions_dim = 1
buffer_size = 10
batch_size = 3
n_actions = 10

memory = ReplayBuffer(
    state_dim=state_dim, actions_dim=actions_dim, buffer_size=buffer_size, batch_size=batch_size, n_actions=n_actions
)
for i in range(1, 13):
    memory.add(array([i, 10 * i]), (i - 1) % buffer_size, i, array([i, 20*i]), False)
    
states, actions, rewards, next_states, dones, actions_oh = memory.sample()

print(states)
print(actions)
print(rewards)
print(next_states)
print(dones)
print(actions_oh)

# +
state_dim = 2
actions_dim = 1
buffer_size = 10
batch_size = 3
n_actions = 10

memory = ReplayBuffer(
    state_dim=state_dim, actions_dim=actions_dim, buffer_size=buffer_size, batch_size=batch_size, n_actions=n_actions
)
for i in range(1, 13):
    memory.add(array([i, 10 * i]), array([(i - 1) % buffer_size]), i, array([i, 20*i]), False)
    
states, actions, rewards, next_states, dones, actions_oh = memory.sample()

print(states)
print(actions)
print(rewards)
print(next_states)
print(dones)
print(actions_oh)

# +
state_dim = 2
actions_dim = 2
buffer_size = 10
batch_size = 3
n_actions = 10

memory = ReplayBuffer(
    state_dim=state_dim, actions_dim=actions_dim, buffer_size=buffer_size, batch_size=batch_size, n_actions=n_actions
)
for i in range(1, 13):
    memory.add(array([i, 10 * i]), array([i, i]), i, array([i, 20*i]), False)
    
states, actions, rewards, next_states, dones, actions_oh = memory.sample()

print(states)
print(actions)
print(rewards)
print(next_states)
print(dones)
print(actions_oh)
# -

# <a name="2-2"></a>
# ## Experience Replay Buffer
# [ToC](#ToC)  
#
# [Article with Explanatin](https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/)  
# [Article](https://python.plainenglish.io/d3qn-agent-with-prioritized-experience-replay-799f6e95264)


# <a name="2-2-1"></a>
# ### Simple Examples
# [ToC](#ToC) 

# +
state_dim = 2
actions_dim = 1
buffer_size = 10
batch_size = 3
n_actions = 10
alpha = 0.2

beta = 0.9

memory = PrioritizedReplayBuffer(
    state_dim=state_dim, actions_dim=actions_dim, buffer_size=buffer_size, batch_size=batch_size, n_actions=n_actions,
    alpha=alpha
)

for i in range(1, 13):
    memory.add(array([i, 10 * i]), (i - 1) % buffer_size, i, array([i, 20*i]), False)
    
states, actions, rewards, next_states, dones, actions_oh, weights, indices  = memory.sample(beta)

print(states)
print(actions)
print(rewards)
print(next_states)
print(dones)
print(actions_oh)
print(weights)
print(indices)

# +
state_dim = 2
actions_dim = 1
buffer_size = 10
batch_size = 3
n_actions = 10
alpha = 0.2

beta = 0.9

memory = PrioritizedReplayBuffer(
    state_dim=state_dim, actions_dim=actions_dim, buffer_size=buffer_size, batch_size=batch_size, n_actions=n_actions,
    alpha=alpha
)

for i in range(1, 13):
    memory.add(array([i, 10 * i]), array([(i - 1) % buffer_size]), i, array([i, 20*i]), False)
    
states, actions, rewards, next_states, dones, actions_oh, weights, indices  = memory.sample(beta)

print(states)
print(actions)
print(rewards)
print(next_states)
print(dones)
print(actions_oh)
print(weights)
print(indices)

# +
state_dim = 2
actions_dim = 2
buffer_size = 10
batch_size = 3
n_actions = 10
alpha = 0.2

beta = 0.9

memory = PrioritizedReplayBuffer(
    state_dim=state_dim, actions_dim=actions_dim, buffer_size=buffer_size, batch_size=batch_size, n_actions=n_actions,
    alpha=alpha
)

for i in range(1, 13):
    memory.add(array([i, 10 * i]), array([i, i]), i, array([i, 20*i]), False)
    
states, actions, rewards, next_states, dones, actions_oh, weights, indices  = memory.sample(beta)

print(states)
print(actions)
print(rewards)
print(next_states)
print(dones)
print(actions_oh)
print(weights)
print(indices)
# -

# <a name="2-2-2"></a>
# ### Examples with Update of Priorities
# [ToC](#ToC) 

# +
state_dim = 2
actions_dim = 2
buffer_size = 10
batch_size = 5
n_actions = 10
alpha = 0.2

beta = 0.9

memory = PrioritizedReplayBuffer(
    state_dim=state_dim, actions_dim=actions_dim, buffer_size=buffer_size, batch_size=batch_size, n_actions=n_actions,
    alpha=alpha
)

for i in range(1, 13):
    memory.add(array([i, 10 * i]), (i - 1) % buffer_size, i, array([i, 20*i]), False)
    
states, actions, rewards, next_states, dones, actions_oh, weights, indices  = memory.sample(beta)

print(indices)
print(weights)

# update
memory.update_priorities(indices, array([2.] * batch_size).reshape((batch_size, 1)))
states, actions, rewards, next_states, dones, actions_oh, weights, indices  = memory.sample(beta)
print(indices)
print(weights)

# update
memory.update_priorities(indices, array([3.] * batch_size).reshape((batch_size, 1)))
states, actions, rewards, next_states, dones, actions_oh, weights, indices  = memory.sample(beta)
print(indices)
print(weights)

# update
memory.update_priorities(indices, array([4.] * batch_size).reshape((batch_size, 1)))
states, actions, rewards, next_states, dones, actions_oh, weights, indices  = memory.sample(beta)
print(indices)
print(weights)
# -

# <a name="3"></a>
# # Final Timestamp
# [ToC](#ToC)  

Logger().end_timer()
