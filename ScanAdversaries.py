import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
import sys
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import matplotlib.colors as colors

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

plt.rcParams.update({'font.family':'cmr10','font.size': 13})
matplotlib.rcParams['axes.unicode_minus']=False
matplotlib.rcParams['axes.labelsize']=15
plt.rcParams['figure.figsize']=(4,4)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True


# ********** Load data *******************************
SignalDF = pd.read_csv('data/BSM_WW-jet_RelaxedMassCut/data_sig.txt',
                       skiprows=2,
                       index_col=False,
                       names=['(pruned)m', 'pT', 'tau_(1)^(1/2)',
                              'tau_(1)^(1)', 'tau_(1)^(2)',
                              'tau_(2)^(1/2)', 'tau_(2)^(1)', 'tau_(2)^(2)',
                              'tau_(3)^(1/2)', 'tau_(3)^(1)', 'tau_(3)^(2)',
                              'tau_(4)^(1)', 'tau_(4)^(2)']
                       )
BackgroundDF = pd.read_csv('data/BSM_WW-jet_RelaxedMassCut/data_bkg.txt',
                           skiprows=2,
                           index_col=False,
                           names=['(pruned)m', 'pT', 'tau_(1)^(1/2)',
                                  'tau_(1)^(1)', 'tau_(1)^(2)',
                                  'tau_(2)^(1/2)', 'tau_(2)^(1)', 'tau_(2)^(2)',
                                  'tau_(3)^(1/2)', 'tau_(3)^(1)', 'tau_(3)^(2)',
                                  'tau_(4)^(1)', 'tau_(4)^(2)']
                           )

columns = SignalDF.columns
labels = [r'$m_j$ [GeV]', r'$p_T$ [GeV]',
          r'$\tau_1^{1/2}$', r'$\tau_1^{1}$',
          r'$\tau_2^{1/2}$', r'$\tau_2^{1}$', r'$\tau_2^{2}$',
          r'$\tau_3^{1/2}$', r'$\tau_3^{1}$', r'$\tau_3^{2}$',
          r'$\tau_4^{1}$', r'$\tau_4^{2}$'
         ]

TrainingColumns = columns

SignalDF['Label'] = 1
BackgroundDF['Label'] = 0

CombinedData = np.vstack([SignalDF[TrainingColumns].values,
                          BackgroundDF[TrainingColumns].values
                          ]
                         )
CombinedLabels = np.hstack([SignalDF['Label'].values,
                            BackgroundDF['Label'].values
                            ]
                           ).reshape(CombinedData.shape[0], 1)

if not os.path.isfile('data/TrainingIndices.npy'):
    indices = np.arange(CombinedLabels.shape[0])
    np.random.shuffle(indices)
    training_size = int(0.7 * len(indices))
    validation_size = int(0.15 * len(indices))
    TrainingIndices = indices[: training_size]
    ValIndices = indices[training_size: training_size + validation_size]
    TestIndices = indices[training_size + validation_size:]

    np.save('data/TrainingIndices.npy', TrainingIndices)
    np.save('data/ValidationIndices.npy', ValidationIndices)
    np.save('data/TestIndices.npy', TestIndices)
else:
    TrainingIndices = np.load('data/TrainingIndices.npy')
    ValIndices = np.load('data/ValidationIndices.npy')
    TestIndices = np.load('data/TestIndices.npy')

X_train, y_train = CombinedData[TrainingIndices], CombinedLabels[TrainingIndices]
X_test, y_test = CombinedData[TestIndices], CombinedLabels[TestIndices]
X_val, y_val = CombinedData[ValIndices], CombinedLabels[ValIndices]

mass_train = X_train[:, 0]
mass_test = X_test[:, 0]
mass_val = X_val[:, 0]

class_weights = {1: float(len(CombinedData)) / len(SignalDF),
                 0: float(len(CombinedData)) / len(BackgroundDF)
                 }

SS = StandardScaler()
X_trainscaled = SS.fit_transform(X_train)
X_testscaled = SS.transform(X_test)
X_valscaled = SS.transform(X_val)
