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

class_weights = {1: float(len(CombinedData)) / len(SignalDF),
                 0: float(len(CombinedData)) / len(BackgroundDF)
                 }

SS = StandardScaler()
X_trainscaled = SS.fit_transform(X_train)
X_testscaled = SS.transform(X_test)
X_valscaled = SS.transform(X_val)

# ****************************************************
# Digitize the masses for the adversary
# ****************************************************
massbins = np.linspace(50, 400, 11)

mbin_train = np.digitize(mass_train, massbins) - 1
mbin_test = np.digitize(mass_test, massbins) - 1
mbin_validate = np.digitize(mass_val, massbins) - 1

mbin_train_labels = keras.utils.to_categorical(mbin_train, num_classes=10)
mbin_test_labels = keras.utils.to_categorical(mbin_test, num_classes=10)
mbin_validate_labels = keras.utils.to_categorical(mbin_validate, num_classes=10)

# ****************************************************
# Network information
# ****************************************************
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1,
                              patience=3, min_lr=1.0e-6)
es = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

inputs = Input(shape=(len(TrainingColumns), ))
Classifier = Dense(512, activation='relu')(inputs)
Classifier = Dense(32, activation='relu')(Classifier)
Classifier = Dense(1, activation='sigmoid')(Classifier)
ClassifierModel = Model(inputs=inputs, outputs=Classifier)

# ***************************************************
# Train the Classifier
# No adversary, just do as good as possible
# ***************************************************
if not os.path.isfile('OriginalClassifer.h5'):
    ClassifierModel.compile(optimizer='adam', loss='binary_crossentropy')
    ClassifierModel.summary()
    ClassifierModel.fit(X_trainscaled,
                        y_train,
                        validation_data=[X_valscaled, y_val],
                        epochs=50,
                        class_weight=class_weights,
                        callbacks=[reduce_lr, es]
                        )

    OriginalPreds = ClassifierModel.predict(X_testscaled)
    fpr_O, tpr_O, thresholds_O = roc_curve(y_test, OriginalPreds)
    auc_O = auc(fpr_O, tpr_O)
    i50 = np.argmin(np.abs(tpr_O - 0.5))
    i60 = np.argmin(np.abs(tpr_O - 0.6))
    i70 = np.argmin(np.abs(tpr_O - 0.7))
    i80 = np.argmin(np.abs(tpr_O - 0.8))
    i90 = np.argmin(np.abs(tpr_O - 0.9))

    fp50, tp50, thr50 = fpr_O[i50], tpr_O[i50], thresholds_O[i50]
    fp60, tp60, thr60 = fpr_O[i60], tpr_O[i60], thresholds_O[i60]
    fp70, tp70, thr70 = fpr_O[i70], tpr_O[i70], thresholds_O[i70]
    fp80, tp80, thr80 = fpr_O[i80], tpr_O[i80], thresholds_O[i80]
    fp90, tp90, thr90 = fpr_O[i90], tpr_O[i90], thresholds_O[i90]

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.hist(OriginalPreds[y_test == 0], histtype='step',
             bins=50, color='k'
             )
    plt.hist(OriginalPreds[y_test == 1], histtype='step',
             bins=50, color='C0'
             )
    plt.plot([], [], color='k', label='Backgrounds')
    plt.plot([], [], color='C0', label='Signals')
    plt.vlines([thr50, thr60, thr70, thr80, thr90],
               ymin=1e-1, ymax=1e3,
               colors=['C5', 'C4', 'C3', 'C2', 'C1']
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
    plt.xlabel('False positve rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right', frameon=False, fontsize=10)
    plt.xscale('log')

    plt.subplot(1, 3, 3)
    back_i = (y_test == 0).flatten()
    plt.hist(mass_test[back_i],
             bins=50, range=(50, 400),
             histtype='step', color='k')
    plt.hist(mass_test[back_i & (OriginalPreds.flatten() > thr50)],
             bins=50, range=(50, 400),
             histtype='step', color='C5')
    plt.hist(mass_test[back_i & (OriginalPreds.flatten() > thr60)],
             bins=50, range=(50, 400),
             histtype='step', color='C4')
    plt.hist(mass_test[back_i & (OriginalPreds.flatten() > thr70)],
             bins=50, range=(50, 400),
             histtype='step', color='C3')
    plt.hist(mass_test[back_i & (OriginalPreds.flatten() > thr80)],
             bins=50, range=(50, 400),
             histtype='step', color='C2')
    plt.hist(mass_test[back_i & (OriginalPreds.flatten() > thr90)],
             bins=50, range=(50, 400),
             histtype='step', color='C1')
    plt.hist(mass_test[(y_test == 1).flatten()].flatten(),
             weights=np.ones(np.sum(y_test == 1)) * 0.15,
             bins=50, range=(50, 400),
             color='C0', alpha=0.2)
    plt.yscale('log')
    plt.xlabel(r'$m_j$ [GeV]')
    plt.ylabel('Events per bin')

    plt.suptitle('No Adversary', y=1.02)
    plt.tight_layout(w_pad=2)
    plt.savefig('Plots/InitialNetworkNoAdversary.pdf', bbox_inches='tight')

    ClassifierModel.save('OriginalClassifer.h5')
else:
    ClassifierModel = load_model('OriginalClassifer.h5')

# ***************************************************
# Add in the Adversary
# Now the adversary uses the whole input, but only takes the output of the classifier
# ***************************************************
Adversary = ClassifierModel(inputs)
Adversary = Dense(50, activation='tanh')(Adversary)
Adversary = Dense(10, activation='softmax')(Adversary)
# Adversary = K.tf.nn.softmax(Adversary)

# The adversary only is supposed to work on the backround events
# feed into it the actual label, so that we can make the loss function be 0
# for the signal events
LabelWeights = Input(shape=(1,))
AdversaryC = concatenate([Adversary, LabelWeights], axis=1)

AdversaryModel = Model(inputs=[inputs, LabelWeights],
                       outputs=AdversaryC
                       )


def Make_loss_A(lam):
    def loss(y_true, y_pred):
        y_pred, l_true = y_pred[:, :-1], y_pred[:, -1]  # prediction and label
        return lam * K.mean(K.categorical_crossentropy(y_true, y_pred) *
                            (1.0 - l_true),  # 0 for signal and 1 for background
                            axis=-1)
    return loss


# ***************************************************
# Let the adversary learn for a while
# ***************************************************
ClassifierModel.trainable = False
AdversaryModel.compile(loss=Make_loss_A(1.0),
                       optimizer=SGD(momentum=0.5, lr=1, decay=1e-5))
AdversaryModel.summary()
AdversaryModel.fit(x=[X_trainscaled, y_train],
                   y=mbin_train_labels,
                   validation_data=[[X_valscaled, y_val],
                                    mbin_validate_labels],
                   epochs=50,
                   callbacks=[reduce_lr, es]
                   )

# Now put the two models together into one model
# With two output, there will need to be two losses
CombinedModel = Model(inputs=[inputs, LabelWeights],
                      outputs=[ClassifierModel(inputs),
                               AdversaryModel([inputs, LabelWeights])
                               ]
                      )
