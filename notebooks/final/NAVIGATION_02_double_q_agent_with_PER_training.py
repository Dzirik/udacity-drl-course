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

# # Double Q Agent with Prioritized Experience Replay
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
#     - [Functions](#2-1)   
#     - [Creating Environment](#2-2)    
#     - [Training Agent](#2-3)
#     - [Plotting Results](#2-4)
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

# from src.utils.notebook_support_functions import create_button, get_notebook_name
from src.utils.logger import Logger
from src.utils.envs import Envs
from src.utils.config import Config
from pandas import options
from IPython.display import display, HTML

# > Constants for overall behaviour.

LOGGER_CONFIG_NAME = "logger_file" # default
PYTHON_CONFIG_NAME = "python_repo" # default
CREATE_BUTTON = False
ADDAPT_WIDTH = False
NOTEBOOK_NAME = "02_double_q_agent_with_PER_training" # get_notebook_name()

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

# +
import gym
from collections import deque
from numpy import mean
from typing import List

from importlib import reload

import matplotlib.pyplot as plt
from IPython.display import clear_output
# -

# <a name="1-4"></a>
# ### Internal Code
# [ToC](#ToC)  
# Code, libraries, classes, functions from within the repository.

# +
from src.data.replay_buffer import ReplayBuffer
from src.models.torch_networks import QNetwork, QNetworkDropout, DuelingQNetwork, DuelingQNetworkDropout
from src.models.agents import DQNAgent, DQNAgentPER

from src.utils.timer import Timer
# -

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

# + tags=["parameters"]
NETWORK = "DUEL-Q" # "Q", "DUEL-Q"
DROPOUT = False

EPS_DECAY = 0.995
GAMMA = 0.99

UPDATE_EVERY_STEP = 4
HARD_UPDATE_EVERY_STEPS = 4
# STEPS_UPDATE = (2, 8) # UPDATE_EVERY_STEP, HARD_UPDATE_EVERY_STEPS

ALPHA = 0.2
BETA_START = 0.6
# -

# #### Notebook Specific Constants
# [ToC](#ToC)  

# +
ENV_ID = "LunarLander-v2" # "LunarLander-v2" "CartPole-v1" 

ACTIONS_DIM = 1
MEMORY_SIZE = 2**14 # 2**14 = 16384
BATCH_SIZE = 64

EPS_START = 1.0
EPS_END = 0.01
# -

2**14

# <a name="2"></a>
# # ANALYSIS
# [ToC](#ToC)  

timer = Timer()

# <a name="2-1"></a>
# ## Functions
# [ToC](#ToC)  


def plot(x_data: List[int], y_data: List[float], plot_title: str, y_label: str) -> None:
    """
    Creats simple plot.
    """
    plt.figure(figsize=(15, 6), dpi=80)
    plt.title(plot_title)
    plt.xlabel("Episode Number")
    plt.ylabel(y_label)
    plt.plot(x_data, y_data, "g", linewidth=2)
    plt.show()


# <a name="2-2"></a>
# ## Creating Environment
# [ToC](#ToC)  

env = gym.make(ENV_ID)

# <a name="2-3"></a>
# ## Training Agent
# [ToC](#ToC)  

net = None
if NETWORK == "Q":
    if DROPOUT:
        net = QNetworkDropout
    else:
        net = QNetwork
if NETWORK == "DUEL-Q":
    if DROPOUT:
        net = DuelingQNetworkDropout
    else:
        net = DuelingQNetwork

# +
agent = DQNAgentPER(env, ACTIONS_DIM, MEMORY_SIZE, BATCH_SIZE, net, GAMMA, ALPHA)

agent.set_optimizing_parameters(update_every_steps=UPDATE_EVERY_STEP, hard_update_every_steps=HARD_UPDATE_EVERY_STEPS, tau=0.001)

# +
n_episodes = 4000
max_steps_in_episode = 1000
eps_start = EPS_START
eps_end = EPS_END
eps_decay = EPS_DECAY
beta_start = BETA_START

# storing data for plotting
episodes_indices = []
epss = []
avg_scores = []
betas = []
# (end) storing data for plotting

scores = []
scores_window = deque(maxlen=100)
eps = eps_start
beta = beta_start
timer.start()
for episode in range(1, n_episodes+1):
    state, _ = env.reset()
    score = 0
    
    for step in range(max_steps_in_episode):    
        action = agent.act(state, eps)
        next_state, reward, done = agent.step(action)
        agent.learn(beta)
    
        state = next_state
        score = score + reward
        if done:
            break
    scores.append(score)
    scores_window.append(score)
    eps = max(eps_end, eps_decay * eps)
    fraction = min(episode / n_episodes, 1.0)
    beta = beta_start + fraction * (1.0 - beta_start)
    
    if episode % (n_episodes / 100) == 0:
        episodes_indices.append(episode)
        epss.append(eps)
        avg_scores.append(mean(scores_window))
        betas.append(beta)
    print(f"## Episode Number: {episode}, Average Score: {mean(scores_window)}", end="\r") 
    if episode % (n_episodes / 50) == 0:
        secs, mins = timer.get_meantime()
        print(f"## Episode Number: {episode}, Average Score: {mean(scores_window)}, Duration[s], [mins]: "
              f"{secs}, {mins}{' '*20}") 
    if mean(scores_window)>=210.0:
        print("\n")
        print(f"## ENVIRONMENT SOLVED ##")
        print(f"  - Number of Episodes: {episode - 100}")
        print(f"  - Average Score: {mean(scores_window)}")
        model_file_name = agent.save_model()
        print(f"  - Model Was Saved Into File: {model_file_name}")
        break
# -

timer.end()

# <a name="2-4"></a>
# ## Plotting the Results
# [ToC](#ToC)  

plot_title = "Average Reward from Last 100 Episodes"
y_label = "Average Reward from Last 100 Episodes"
plot(episodes_indices, avg_scores, plot_title, y_label)

plot_title = "Epsilon Value Evolution"
y_label = "Epsilon Value"
plot(episodes_indices, epss, plot_title, y_label)

plot_title = "Beta Value Evolution"
y_label = "Beta Value"
plot(episodes_indices, betas, plot_title, y_label)

# <a name="3"></a>
# # Final Timestamp
# [ToC](#ToC)  

Logger().end_timer()
