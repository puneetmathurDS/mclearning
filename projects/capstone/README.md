# Machine Learning Engineer Nanodegree
## Specializations
## Project: Capstone Proposal and Capstone Project

The following Python version 2.7 standard libraries have been used in the project.
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score

The dataset is 1+GB csv file and can be downloaded from the following public url:
http://www.PuneetMathur.in/fd2kfullfinal.csv

First run the Capstone.ipynb after downloading the csv dataset file.
You may change the csv file path in the following Capstone.ipynb file:

data = pd.read_csv("fd2kfullfinal.csv", low_memory=False)

also below if it does not work for you:
plt.savefig('exploratory.png')

Rest is a simple and not extra libraries apart from the ones mentioned above should be able to run on a Python 2.7 machine.

