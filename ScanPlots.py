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

if not os.path.isfile('data/TrainingIndices.npy'):
    indices = np.arange(CombinedLabels.shape[0])
    np.random.shuffle(indices)
    training_size = int(0.7 * len(indices))
    validation_size = int(0.15 * len(indices))
    TrainingIndices = indices[: training_size]
    ValIndices = indices[training_size: training_size + validation_size]
    TestIndices = indices[training_size + validation_size:]

    np.save('data/TrainingIndices.npy', TrainingIndices)
    np.save('data/ValidationIndices.npy', ValIndices)
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

X_train = X_train[:, 1:]
X_test = X_test[:, 1:]
X_val = X_val[:, 1:]

class_weights = {1: float(len(CombinedData)) / len(SignalDF),
                 0: float(len(CombinedData)) / len(BackgroundDF)
                 }

val_weights = np.ones_like(y_val)
val_weights[y_val == 0] = class_weights[0]
val_weights[y_val == 1] = class_weights[1]
val_weights = val_weights.flatten()

tr_weights = np.ones_like(y_train)
tr_weights[y_train == 0] = class_weights[0]
tr_weights[y_train == 1] = class_weights[1]
tr_weights = tr_weights.flatten()

SS = StandardScaler()
X_trainscaled = SS.fit_transform(X_train)
X_testscaled = SS.transform(X_test)
X_valscaled = SS.transform(X_val)

# ****************************************************
# Digitize the masses for the adversary
# ****************************************************
mass_bins_setup = mass_train.copy()
mass_bins_setup = mass_bins_setup[(y_train == 0).flatten()]
mass_bins_setup.sort()
size = int(len(mass_bins_setup) / 10)

massbins = [50,
            mass_bins_setup[size], mass_bins_setup[size * 2],
            mass_bins_setup[size * 3], mass_bins_setup[size * 4],
            mass_bins_setup[size * 5], mass_bins_setup[size * 6],
            mass_bins_setup[size * 7], mass_bins_setup[size * 8],
            mass_bins_setup[size * 9],
            400]

print(massbins)

mbin_train = np.digitize(mass_train, massbins) - 1
mbin_test = np.digitize(mass_test, massbins) - 1
mbin_validate = np.digitize(mass_val, massbins) - 1

mbin_train_labels = keras.utils.to_categorical(mbin_train, num_classes=10)
mbin_test_labels = keras.utils.to_categorical(mbin_test, num_classes=10)
mbin_validate_labels = keras.utils.to_categorical(mbin_validate, num_classes=10)


# ******************************
# Network Setup
# ******************************
inputs = Input(shape=(X_test.shape[1], ))
Classifier = Dense(50, activation='relu')(inputs)
Classifier = Dense(50, activation='relu')(Classifier)
Classifier = Dense(50, activation='relu')(Classifier)
Classifier = Dense(1, activation='sigmoid')(Classifier)
ClassifierModel = Model(inputs=inputs, outputs=Classifier)

# *****************************
# Scan
# *****************************
for i in range(12):
    lam = 10**i
    ClassifierModel.load_weights('Models/Class_lam_{0}_final_weights.h5'.format(lam))

    ClassifierModel.compile(optimizer='adam', loss='binary_crossentropy')
    FinalPreds = ClassifierModel.predict(X_testscaled)
    fpr_O, tpr_O, thresholds_O = roc_curve(y_test, FinalPreds)
    auc_O = auc(fpr_O, tpr_O)
    i50 = np.argmin(np.abs(tpr_O - 0.5))
    i60 = np.argmin(np.abs(tpr_O - 0.6))
    i70 = np.argmin(np.abs(tpr_O - 0.7))
    i80 = np.argmin(np.abs(tpr_O - 0.8))
    i90 = np.argmin(np.abs(tpr_O - 0.9))
    i95 = np.argmin(np.abs(tpr_O - 0.95))

    fp50, tp50, thr50 = fpr_O[i50], tpr_O[i50], thresholds_O[i50]
    fp60, tp60, thr60 = fpr_O[i60], tpr_O[i60], thresholds_O[i60]
    fp70, tp70, thr70 = fpr_O[i70], tpr_O[i70], thresholds_O[i70]
    fp80, tp80, thr80 = fpr_O[i80], tpr_O[i80], thresholds_O[i80]
    fp90, tp90, thr90 = fpr_O[i90], tpr_O[i90], thresholds_O[i90]
    fp95, tp95, thr95 = fpr_O[i95], tpr_O[i95], thresholds_O[i95]

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.hist(FinalPreds[y_test == 0], histtype='step',
             bins=50, color='k'
             )
    plt.hist(FinalPreds[y_test == 1], histtype='step',
             bins=50, color='C0'
             )
    plt.plot([], [], color='k', label='Backgrounds')
    plt.plot([], [], color='C0', label='Signals')
    plt.vlines([thr50, thr60, thr70, thr80, thr90, thr95],
               ymin=1e-1, ymax=1e3,
               colors=['C5', 'C4', 'C3', 'C2', 'C1', 'C6']
               )
    plt.yscale('log')
    plt.legend(loc='upper left', frameon=False, fontsize=10)
    plt.xlabel('Network output')
    plt.ylabel('Events per bin')
    plt.ylim(1e-1, 1e5)

    plt.subplot(1, 3, 2)
    plt.plot(fpr_O, tpr_O, label='early stopping: {0:.2f}'.format(auc_O))
    plt.scatter(fp50, tp50, color='C5')
    plt.scatter(fp60, tp60, color='C4')
    plt.scatter(fp70, tp70, color='C3')
    plt.scatter(fp80, tp80, color='C2')
    plt.scatter(fp90, tp90, color='C1')
    plt.scatter(fp95, tp95, color='C6')
    plt.xlabel('False positve rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right', frameon=False, fontsize=10)
    plt.xscale('log')

    plt.subplot(1, 3, 3)
    back_i = (y_test == 0).flatten()
    plt.hist(mass_test[~back_i],
             weights=np.ones(np.sum(y_test == 1)) * 0.15,
             bins=50, range=(50, 400),
             color='C0', alpha=0.5)
    plt.hist(mass_test[back_i],
             bins=50, range=(50, 400),
             histtype='step', color='k')
    plt.hist(mass_test[back_i & (FinalPreds.flatten() > thr50)],
             bins=50, range=(50, 400),
             histtype='step', color='C5')
    plt.hist(mass_test[back_i & (FinalPreds.flatten() > thr60)],
             bins=50, range=(50, 400),
             histtype='step', color='C4')
    plt.hist(mass_test[back_i & (FinalPreds.flatten() > thr70)],
             bins=50, range=(50, 400),
             histtype='step', color='C3')
    plt.hist(mass_test[back_i & (FinalPreds.flatten() > thr80)],
             bins=50, range=(50, 400),
             histtype='step', color='C2')
    plt.hist(mass_test[back_i & (FinalPreds.flatten() > thr90)],
             bins=50, range=(50, 400),
             histtype='step', color='C1')
    plt.hist(mass_test[back_i & (FinalPreds.flatten() > thr95)],
             bins=50, range=(50, 400),
             histtype='step', color='C6')
    plt.yscale('log')
    plt.xlabel(r'$m_j$ [GeV]')
    plt.ylabel('Events per bin')

    plt.suptitle('Adversary $\lambda=$' + str(lam), y=1.02)
    plt.tight_layout(w_pad=2)
    plt.savefig('Plots/Lambda_{0}/Adversary_lam_{0}_final.pdf'.format(lam),
                bbox_inches='tight')
