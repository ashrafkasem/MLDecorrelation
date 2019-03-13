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

TrainingIndices = np.load('data/TrainingIndices.npy')
ValIndices = np.load('data/ValidationIndices.npy')
TestIndices = np.load('data/TestIndices.npy')

# ********** Load data *******************************
SignalDF = pd.read_csv('data/data_sig.txt',
                       skiprows=2,
                       index_col=False,
                       names=['(pruned)m', 'pT', 'tau_(1)^(1/2)',
                              'tau_(1)^(1)', 'tau_(1)^(2)',
                              'tau_(2)^(1/2)', 'tau_(2)^(1)', 'tau_(2)^(2)',
                              'tau_(3)^(1/2)', 'tau_(3)^(1)', 'tau_(3)^(2)',
                              'tau_(4)^(1)', 'tau_(4)^(2)']
                       )
BackgroundDF = pd.read_csv('data/data_bkg.txt',
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

X_train, y_train = CombinedData[TrainingIndices], CombinedLabels[TrainingIndices]
X_test, y_test = CombinedData[TestIndices], CombinedLabels[TestIndices]
X_val, y_val = CombinedData[ValIndices], CombinedLabels[ValIndices]

mass_train = X_train[:, 0]
mass_test = X_test[:, 0]
mass_val = X_val[:, 0]

# Do not use the mass for classification
X_train = X_train[:, 2:]
X_test = X_test[:, 2:]
X_val = X_val[:, 2:]

class_weights = {1: float(len(CombinedData)) / len(SignalDF),
                 0: float(len(CombinedData)) / len(BackgroundDF)
                 }

SS = StandardScaler()
X_trainscaled = SS.fit_transform(X_train)
X_testscaled = SS.transform(X_test)
X_valscaled = SS.transform(X_val)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1,
                              patience=5, min_lr=1.0e-6)
es = EarlyStopping(monitor='val_loss', patience=11, verbose=0, mode='auto')

Histories = {}
Metrics = {}
MAXDEPTH = 9
DIRNAME = 'NS100_noPT'
for depth in range(1, MAXDEPTH):
    print('Working on model with {0} hidden layers with 100 nodes each'.format(depth))
    model = Sequential()
    model.add(Dense(100, input_dim=X_trainscaled.shape[1], activation='relu'))
    for _ in range(depth):
        model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(lr=1e-3),
                  loss='binary_crossentropy')

    val_weights = np.ones_like(y_val)
    val_weights[y_val == 0] = class_weights[0]
    val_weights[y_val == 1] = class_weights[1]
    val_weights = val_weights.flatten()

    history = model.fit(x=X_trainscaled,
                        y=y_train,
                        validation_data=[X_valscaled,
                                         y_val,
                                         val_weights
                                         ],
                        epochs=100,
                        class_weight=class_weights,
                        callbacks=[reduce_lr, es]
                        )
    Histories[depth] = history
    if not os.path.isdir('Models/ArchTest/' + DIRNAME):
        os.mkdir('Models/ArchTest/' + DIRNAME)
        os.mkdir('Models/ArchTest/' + DIRNAME + '/fprtpr')
    if not os.path.isdir('Plots/ArchTest/' + DIRNAME):
        os.mkdir('Plots/ArchTest/' + DIRNAME)
    model.save('Models/ArchTest/' + DIRNAME + '/Depth_{0}.h5'.format(depth))

    OriginalPreds = model.predict(X_testscaled)
    fpr_O, tpr_O, thresholds_O = roc_curve(y_test, OriginalPreds)
    np.save('Models/ArchTest/NS100_noPT/fprtpr/fpr_{0}.npy'.format(depth), fpr_O)
    np.save('Models/ArchTest/NS100_noPT/fprtpr/tpr_{0}.npy'.format(depth), tpr_O)
    auc_O = auc(fpr_O, tpr_O)
    Metrics[depth] = [fpr_O, tpr_O, thresholds_O, auc_O]

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(Histories[depth].history['loss'], label='training data')
    plt.plot(Histories[depth].history['val_loss'], label='validation data')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best', frameon=False, fontsize=12)

    plt.subplot(1, 2, 2)
    plt.plot(tpr_O,
             [1.0 / f for f in fpr_O],
             label='{0} hidden layers: {1:0.04f}'.format(depth, auc_O)
             )
    plt.xlabel(r'$\epsilon_S$')
    plt.ylabel(r'$1 / \epsilon_B$')
    plt.legend(loc='best', frameon=False, fontsize=12)
    plt.yscale('log')

    plt.suptitle('100 nodes per hidden layer', y=1.03, fontsize=16)
    plt.grid()
    plt.minorticks_on()
    plt.ylim(1, 1e4)
    plt.xlim(0, 1)
    plt.tight_layout(w_pad=2)
    plt.savefig('Plots/ArchTest/NS100_noPT/Single_{0}.pdf'.format(depth), bbox_inches='tight')
    plt.close()
    plt.clf()


plt.figure(figsize=(8, 4))
for depth in range(1, MAXDEPTH):
    plt.subplot(1, 2, 1)
    plt.plot(Histories[depth].history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best', frameon=False, fontsize=12)

    fpr_O, tpr_O, thresholds_O, auc_O = Metrics[depth]
    plt.subplot(1, 2, 2)
    plt.plot(tpr_O,
             [1.0 / f for f in fpr_O],
             label='{0} hidden layers: {1:0.04f}'.format(depth, auc_O)
             )
plt.xlabel(r'$\epsilon_S$')
plt.ylabel(r'$1 / \epsilon_B$')
plt.yscale('log')
plt.ylim(1, 1e4)
plt.xlim(0, 1)
plt.grid()
plt.minorticks_on()
plt.legend(loc='best', frameon=False, fontsize=12)
plt.tight_layout(w_pad=2)

plt.suptitle('100 nodes per hidden layer', y=1.03, fontsize=16)
plt.savefig('Plots/ArchTest/NS100_noPT/Combined.pdf', bbox_inches='tight')
plt.close()
plt.clf()
