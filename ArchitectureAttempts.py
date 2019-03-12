import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from subprocess import call

from sklearn import metrics
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import keras
import keras.backend as K
from keras import regularizers
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Input, Concatenate, concatenate
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

plt.rcParams.update({'font.family': 'cmr10',
                     'font.size': 12})
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
