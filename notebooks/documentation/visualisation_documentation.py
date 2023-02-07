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

# # Visualisation Documentation
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
# - [Visualizations Examples](#2)   
#     - [Histogram](#2-1)   
#     - [Bar Chart](#2-2) 
#         - [Bar Chart Sorted by Values](#2-2-1)
#         - [Bar Chart Sorted by IDs](#2-2-2)
#     - [Time Series Visualizations](#2-3)
#         - [Selectors](#2-3-1)
#         - [Time Series](#2-3-2)
#             - [One Time Series](#2-3-2-1)
#             - [Multiple Series](#2-3-2-2)
#             - [Filling](#2-3-2-3)
#             - [Anomalies](#2-3-2-4)
#         - [Time Series Events](#2-3-3)
#             - [Simple Plot](#2-3-3-1)
#             - [Full Plot](#2-3-3-2)
#     - [Multi Histogram - Distplot](#2-4)
#     - [Multi Histogram](#2-5)
#     - [Line Plot](#2-6)
# - [Overall Customisation](#3)
#     - [Size of the Picture](#3-1)
# - [Final Timestamp](#10)

# <a name="0"></a>
# # Notebook Description
# [ToC](#ToC)

# This notebook serves as a documentation notebook for visualisation code from *src\visualisations*.

# <a name="1"></a>
# # GENERAL SETTINGS
# [ToC](#ToC) 
#
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
from src.utils.config import Config
from pandas import options
from IPython.display import display, HTML

# > Constants for overall behaviour.

LOGGER_CONFIG_NAME = "logger_file_console" # default
PYTHON_CONFIG_NAME = "python_local" # default
CREATE_BUTTON = False
ADDAPT_WIDTH = True
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

from importlib import reload
from pandas import Series, options, DataFrame
from numpy.random import seed, normal, randint
from numpy import array, log
from numpy.random import randn
from datetime import datetime

# <a name="1-4"></a>
# ### Internal Code
# [ToC](#ToC)   
# Code, libraries, classes, functions from within the repository.

from src.visualisations.visualisation_functions import create_time_series

# <a name="1-5"></a>
# ### Constants
# [ToC](#ToC)   
# Constants for the notebook.
#
# > *NOTE: Please use all letters upper.*

# #### General Constants
# [ToC](#ToC)   

# from src.global_constants import *  # Remember to import only the constants in use
N_ROWS_TO_DISPLAY = 2
FIGURE_SIZE_SETTING = {"autosize": False, "width": 2200, "height": 750}
CRY_CONFIG_NAME = "cry_basic"

# #### Constants for Setting Automatic Run
# [ToC](#ToC)   



# #### Notebook Specific Constants
# [ToC](#ToC)   

ATTR_OPEN = "OPEN"
ATTR_HIGH = "HIGH"
ATTR_LOW = "LOW"
ATTR_CLOSE = "CLOSE"

# <a name='2'></a>
# # Visualizations Examples
# [ToC](#ToC) 

# <a name='2-1'></a>
# ## Histogram
# [ToC](#ToC) 


# +
seed(185)

data = normal(size=500)
print(f"Type of the data is: {type(data)}")
print(f"Several observations: {data[0:10]}")

plot_title = "My Super Random Histogram"
x_title = "Awesome Random Data"
# -

# ### Plane Histogram
# [ToC](#ToC) 

# +
import src.visualisations.plotly_histogram as HIST

reload(HIST)

histogram = HIST.PlotlyHistogram()

histogram.plot(data=data, plot_title=plot_title, x_title=x_title)
# -

# ### Number of Bins
# [ToC](#ToC) 

# +
import src.visualisations.plotly_histogram as HIST

reload(HIST)

histogram = HIST.PlotlyHistogram()

histogram.plot(data=data, plot_title=plot_title, x_title=x_title, n_bins=4)
# -

# ### Setting the X-axis Range
# [ToC](#ToC) 

# +
import src.visualisations.plotly_histogram as HIST

reload(HIST)

histogram = HIST.PlotlyHistogram()

histogram.plot(data=data, plot_title=plot_title, x_title=x_title, n_bins=0, x_axis_min_max=(-6, 6))
# -

# ### Set Vertical Line
# [ToC](#ToC) 

# +
import src.visualisations.plotly_histogram as HIST

reload(HIST)

histogram = HIST.PlotlyHistogram()

histogram.plot(data=data, plot_title=plot_title, x_title=x_title, n_bins=0, x_axis_min_max=(-6, 6), 
               vertical_lines_positions=[-1.7, 0.4, 2.33])
# -

# <a name='2-2'></a>
# ## Bar Chart
# [ToC](#ToC) 

# <a name='2-2-1'></a>
# ## Bar Chart Sorted by Values
# [ToC](#ToC) 

# +
array_id = array(["Alpha", "Beta", "Gamma"])
array_values = array([10, 5, 30])
plot_title = "Super Bar Chart"
name_id = "Greek Alphabet"
name_values = "Occurance"

order_by_values = True

# +
import src.visualisations.plotly_bar_chart as BAR

reload(BAR)

bar_chart = BAR.PlotlyBarChart()

bar_chart.plot(array_id, array_values, plot_title, name_id, name_values, order_by_values)
# -

# <a name='2-2-2'></a>
# ## Bar Chart Sorted by IDs
# [ToC](#ToC) 

# +
array_id = array(["Alpha", "Beta", "Gamma"])
array_values = array([10, 5, 30])
plot_title = "Super Bar Chart"
name_id = "Greek Alphabet"
name_values = "Occurance"

order_by_values = False

# +
import src.visualisations.plotly_bar_chart as BAR

reload(BAR)

bar_chart = BAR.PlotlyBarChart()

bar_chart.plot(array_id, array_values, plot_title, name_id, name_values, order_by_values)

# +
array_id = array([3, 2, 4, 1])
array_values = array([10, 5, 7, 30])
plot_title = "Super Bar Chart"
name_id = "Greek Alphabet"
name_values = "Occurance"

order_by_values = False

# +
import src.visualisations.plotly_bar_chart as BAR

reload(BAR)

bar_chart = BAR.PlotlyBarChart()

bar_chart.plot(array_id, array_values, plot_title, name_id, name_values, order_by_values)
# -

# <a name='2-3'></a>
# ## Time Series Visualization
# [ToC](#ToC) 

# +
ts_1 = create_time_series(seed_number=11, x_multiplier=0.3)
ts_1_names = [f"Amazing Data {i}" for i in range(1, len(ts_1)+1)]
ts_2 = create_time_series(seed_number=32, x_multiplier=1.1)
ts_2_names = [f"Random Data {i}" for i in range(1, len(ts_2)+1)]
ts_3 = create_time_series(seed_number=64, x_multiplier=1.8)
ts_3_names = [f"Third TS {i}" for i in range(1, len(ts_3)+1)]

# ts_4 = create_time_series(seed_number=20, x_multiplier=0.7)
# ts_5 = create_time_series(seed_number=87, x_multiplier=0.1)
# ts_6 = create_time_series(seed_number=63, x_multiplier=2.5)

print(type(ts_1))
print(type(ts_1[0]))
print(type(ts_1.index))
print("\n")
print(ts_1.head())

plot_title = "Awesome Random Time Series Plot"
y_title = "Awesome Random Data"
# -

# <a name='2-3-1'></a>
# ### Selectors
# [ToC](#ToC) 

# +
import src.visualisations.plotly_time_series as TS

reload(TS)

ts_vizu_selectors = TS.PlotlyTimeSeries()

ts_vizu_selectors.set_selectors(selectors=[
    {"count": 3, "label": "3d", "step": "day", "stepmode": "backward"},
    {"count": 7, "label": "1w", "step": "day", "stepmode": "backward"},
    {"step": "all"}
])

ts_vizu_selectors.plot(
    series=[ts_1], 
    anomalies=[2, 10],
    plot_title=plot_title, 
    y_title=y_title
)
# -

# <a name='2-3-2'></a>
# ### Time Series
# [ToC](#ToC) 

# <a name='2-3-2-1'></a>
# #### One Time Series
# [ToC](#ToC) 

# +
import src.visualisations.plotly_time_series as TS

reload(TS)

ts_visu = TS.PlotlyTimeSeries()

ts_visu.plot(
    series=[ts_1],
    plot_title=plot_title, 
    y_title=y_title
)
# -

# <a name='2-3-2-2'></a>
# #### Multiple Series
# [ToC](#ToC) 

# +
import src.visualisations.plotly_time_series as TS

reload(TS)

ts_visu = TS.PlotlyTimeSeries()

ts_visu.plot(
    series=[ts_1, ts_2, ts_3],  
    series_names=["Random Sinus", "Another Random Sinus", "Wooow"],
    series_obs_names=[ts_1_names, ts_2_names, ts_3_names],
    plot_title=plot_title, 
    y_title=y_title
)
# -

# <a name='2-3-2-3'></a>
# #### Filling
# [ToC](#ToC) 

# +
import src.visualisations.plotly_time_series as TS

reload(TS)

ts_visu = TS.PlotlyTimeSeries()

ts_visu.plot(
    series=[ts_1, ts_1+1, ts_1-2], 
    plot_title=plot_title, 
    y_title=y_title, 
    fill_areas=True
)
# -

# <a name='2-3-2-4'></a>
# #### Anomalies
# [ToC](#ToC) 

# +
import src.visualisations.plotly_time_series as TS

reload(TS)

ts_visu = TS.PlotlyTimeSeries()

ts_visu.plot(
    series=[ts_1, ts_2],  
    series_names=["Random Sinus", "Another Random Sinus"],
    series_obs_names=[ts_1_names, ts_2_names],
    anomalies=[2, 10],
    anomalies_obs_names=["False", "Perfect"],
    plot_title=plot_title, 
    y_title=y_title
)
# -

# <a name='2-3-3'></a>
# ### Time Series Events
# [ToC](#ToC) 

# +
ts_1 = create_time_series(seed_number=11, x_multiplier=0.3)
ts_1_names = [f"Amazing Data {i}" for i in range(1, len(ts_1)+1)]
ts_2 = create_time_series(seed_number=32, x_multiplier=1.1)
ts_2_names = [f"Random Data {i}" for i in range(1, len(ts_2)+1)]

buys = ts_1.iloc[[1, 9, 10, 12, 13, 14],]
buy_names = [f"Buy {i+1}" for i, _ in enumerate(buys)]
sells = ts_1.iloc[[4, 17, 20, 22, 26],]
sell_names = [f"Sell {i+1}" for i, _ in enumerate(sells)]

print(type(ts_1))
print(type(ts_1[0]))
print(type(ts_1.index))
print("\n")
print(ts_1.head())

plot_title = "Awesome Random Time Series Plot"
y_title = "Awesome Random Data"
# -

# <a name='2-3-3-1'></a>
# #### Simple Plot
# [ToC](#ToC) 

# +
import src.visualisations.plotly_time_series_events as TS

reload(TS)

ts_visu = TS.PlotlyTimeSeriesEvents()

ts_visu.plot(
    series=[ts_1],
    events=[buys, sells],
    plot_title=plot_title, 
    y_title=y_title
)
# -

# <a name='2-3-3-2'></a>
# #### Full Plot
# [ToC](#ToC) 

# +
import src.visualisations.plotly_time_series_events as TS

reload(TS)

ts_visu = TS.PlotlyTimeSeriesEvents()

ts_visu.plot(
    series=[ts_1, ts_2],
    series_names=["Wonder Series", "Normal Series"],
    series_obs_names=[ts_1_names, ts_2_names],
    events=[buys, sells],
    event_names=["Buys", "Sells"],
    event_obs_names=[buy_names, sell_names],
    plot_title=plot_title, 
    y_title=y_title
)
# -

# <a name='2-4'></a>
# ## Multi Histogram - Distplot
# [ToC](#ToC) 
#
# The visualization normalizes the histograms, so **if the number of observations is different, it does not work**.

# +
n = 100

x1 = randn(n)-2
x2 = randn(n+20)
x3 = randn(n+100)+2
x4 = randn(n+200)+4
x5 = randn(n+200)+6
x6 = randn(n+200)+8

data = [x1, x2, x3, x4, x5, x6]

group_labels = ["My Group 1", "My Group 2", "My Group 3", "My Group 4", "My Group 5", "My Group 6"]
bin_size = [0.1, 0.25, 0.5, 1, 0.125, 0.75]

plot_title = "My Super Random Histogram"
x_title = "Awesome Random Data"
# -

# ### Basic Distplot
# [ToC](#ToC) 

# +
import src.visualisations.plotly_histogram_distplot as HIST_DIST

reload(HIST_DIST)

histogram_distplot = HIST_DIST.PlotlyHistogramDistplot()

histogram_distplot.plot(data=data, plot_title=plot_title, x_title=x_title, group_labels=group_labels, bin_size=bin_size,
                     x_axis_min_max=None, vertical_lines_positions=None, dashboard=False)
# -

# ### Displot with Range
# [ToC](#ToC) 

# +
import src.visualisations.plotly_histogram_distplot as HIST_DIST

reload(HIST_DIST)

histogram_distplot = HIST_DIST.PlotlyHistogramDistplot()

histogram_distplot.plot(data=data, plot_title=plot_title, x_title=x_title, group_labels=group_labels, bin_size=bin_size,
                     x_axis_min_max=[0, 4], vertical_lines_positions=None, dashboard=False)
# -

# ### Distplot with Lines
# [ToC](#ToC) 

# +
import src.visualisations.plotly_histogram_distplot as HIST_DIST

reload(HIST_DIST)

histogram_distplot = HIST_DIST.PlotlyHistogramDistplot()

histogram_distplot.plot(data=data, plot_title=plot_title, x_title=x_title, group_labels=group_labels, bin_size=bin_size,
                     x_axis_min_max=None, vertical_lines_positions=[-1.7, 0.75, 6.74], dashboard=False)
# -

# <a name='2-5'></a>
# ## Multi Histogram
# [ToC](#ToC) 

# +
n = 100

x1 = randn(n)-2
x2 = randn(n+20)
x3 = randn(n+100)+2
x4 = randn(n+200)+4
x5 = randn(n+200)+6
x6 = randn(n+200)+8

data = [x1, x2, x3, x4, x5, x6]

group_labels = ["My Group 1", "My Group 2", "My Group 3", "My Group 4", "My Group 5", "My Group 6"]

plot_title = "My Super Random Histogram"
x_title = "Awesome Random Data"
# -

# ### Basic Multi Histogram
# [ToC](#ToC) 

# +
import src.visualisations.plotly_histogram_multi as HIST_MULTI

reload(HIST_MULTI)

histogram_multi = HIST_MULTI.PlotlyHistogramMulti()

histogram_multi.plot(data=data, plot_title=plot_title, x_title=x_title, group_labels=group_labels)
# -
# ### Full Version
# [ToC](#ToC) 


# +
import src.visualisations.plotly_histogram_multi as HIST_MULTI

reload(HIST_MULTI)

histogram_multi = HIST_MULTI.PlotlyHistogramMulti()

histogram_multi.plot(data=data, plot_title=plot_title, x_title=x_title, group_labels=group_labels, 
                    x_axis_min_max=(-7, 15), vertical_lines_positions=[-2, 0.5, 5.8])
# -

# <a name="2-6"></a>
# ## Line Plot
# [ToC](#ToC) 

x_list = list(range(1, 26))
x = array(x_list)
y_1 = array([0.5 * x for x in x_list])
y_2 = array([log(x) for x in x_list])
x_short = array(list(range(5, 16)))
y_short = x_short * 2

line_names = ["Linear", "Logarithm", "Shorter Line"]
plot_title = "Line Chart"
x_title = "X Values"
y_title = "Y Values"

# ### Basic Version
# [ToC](#ToC) 

# +
import src.visualisations.plotly_line_chart as LINE

reload(LINE)

line_chart = LINE.PlotlyLineChart()

line_chart.plot(lines=[(x, y_1), (x, y_2)], plot_title=plot_title, x_title=x_title, y_title=y_title)
# -

# ### Full Version
# [ToC](#ToC) 

# +
import src.visualisations.plotly_line_chart as LINE

reload(LINE)

line_chart = LINE.PlotlyLineChart()

line_chart.plot(lines=[(x, y_1), (x, y_2), (x_short, y_short)], line_names=line_names, plot_title=plot_title, x_title=x_title, y_title=y_title)
# -

# <a name="3"></a>
# # Overall Customisation
# [ToC](#ToC) 

# <a name="3-1"></a>
# ## Size of the Picture
# [ToC](#ToC) 

# +
seed(185)

data = normal(size=500)
print(f"Type of the data is: {type(data)}")
print(f"Several observations: {data[0:10]}")

plot_title = "My Super Random Histogram"
x_title = "Awesome Random Data"

# +
import src.visualisations.plotly_histogram as HIST

reload(HIST)

histogram = HIST.PlotlyHistogram()

histogram.plot(data=data, plot_title=plot_title, x_title=x_title)

# +
import src.visualisations.plotly_histogram as HIST

reload(HIST)

histogram = HIST.PlotlyHistogram()
histogram.customize_size(autosize=False, width=2000, height=1500)

histogram.plot(data=data, plot_title=plot_title, x_title=x_title)
# -

# <a name="10"></a>
# # Final Timestamp
# [ToC](#ToC) 

Logger().end_timer()
