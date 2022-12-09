########### IMPORT NECESSARY PYTHON LIBRARIES ##########

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import datetime
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets


import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

########### NAME OF TRAINED AND SAVED MODEL ON KMNIST ##########
mainmod = 'md_kmni.h5'

################################### DATA PROCESSING ################################
x_train = np.load('Data_kmnist/kuzushiji/kmnist-train-imgs.npz')['arr_0']
x_test2 = np.load('Data_kmnist/kuzushiji/kmnist-test-imgs.npz')['arr_0']
y_train = np.load('Data_kmnist/kuzushiji/kmnist-train-labels.npz')['arr_0']
y_test = np.load('Data_kmnist/kuzushiji/kmnist-test-labels.npz')['arr_0']

xtr = np.load('Data_kmnist/kuzushiji/kmnist-train-imgs.npz')['arr_0']
xts = np.load('Data_kmnist/kuzushiji/kmnist-test-imgs.npz')['arr_0']
ytr = np.load('Data_kmnist/kuzushiji/kmnist-train-labels.npz')['arr_0']
yts = np.load('Data_kmnist/kuzushiji/kmnist-test-labels.npz')['arr_0']


###################### INITIALIZATIONS FOR DATA, MODELS AND UQ METHODS #####################

#### DATA PARAMETERS
dim = 28
dim2 = 224
split = 1
ch = 1
n_classes = 10
dty1 = 1
dty2 = 2

#### MAIN MODEL PARAMETERS
batch_sz = 32
epochs = 20
layer_list = [1024, n_classes]

# ENSEMBLE PARAMETERS:
ens = 10
epochs3 = 35

# QIPF PARAMETERS:
bwf = 80 # Silverman Bandwidth multiplier
orr = 12 # No. of moments = (orr/ttk) - 2
ttk = 2 # Choose ever order moments


# TEST DATA LENGTH:
lt = 1000
###################### ------------------------------------------------ #####################



################################### DATA PROCESSING ################################
datx = x_test2[0:lt]
datx = datx.astype('float32')
datx /= 255

y_test = y_test[0:lt]
xts = xts[0:lt]
yts = yts[0:lt]

# ADD NEW AXIS
x_train = x_train[:, :, :, np.newaxis]
datx = datx[:, :, :, np.newaxis]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(datx.shape[0], 'test samples')
print(x_train[0].shape, 'image shape')


num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# DATA NORMALIZATION
x_train = x_train.astype('float32')
x_test = datx.astype('float32')
x_train /= 255
x_test /= 255

x_train, y_train = shuffle(x_train, y_train)
daty = [yts]
datyc = [y_test]

###### VALIDATION DATA FOR TEMPERATURE SCALING UQ METHOD #####
val_tr = x_train[int(0.7 * len(x_train)):len(x_train)]
val_l = y_train[int(0.7 * len(y_train)):len(y_train)]

###################### ------------------------------------------------ #####################




###################################### MODEL PREP #################################################
############### MODEL ARCHITECTURE INITIALIZATION #############
from Models.lenet import LeNet
from Models.resnet import rnet
model = LeNet(x_train[0].shape, n_classes)

# TIMESTAMPED STORAGE OF TRAINING LOGS
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


##################### MODEL TRAINING #########################
# model.fit(x_train, y=y_train,
#           epochs=epochs,
#           validation_data=(val_tr, val_l),
#           callbacks=[TqdmCallback(verbose=0)])

model.load_weights('Models/' + mainmod)
model.summary()
###################### ------------------------------------------------ #####################


################################# IMPORT FUNCTIONS ################################
from UQ_methods import uncq  # UQ IMPLEMENTATIONS
from performance_metrics import perf         # PERFORMANCE QUANTIFICATION OF UQ


############ CALL UQ FUNCTIONS ############

sm8, mm8 = uncq.ensemble(x_train, y_train, datx, epochs3, ens) #### ENSEMBLE ####

sm55, sq55, hmt, tst = uncq.qipf(model, datx, daty[0], bwf, x_train, orr, ttk) #### QIPF ####


################## EVALUATE AVERAGE OF QIPF MODES AT EACH SAMPLE OUTPUT (sm5) ##################
mm5 = np.zeros((len(datx), np.shape(sm55[0])[1]))
mq5 = np.zeros((len(datx), np.shape(sm55[0])[1]))
for i in range(len(sm55)):
    for j in range(np.shape(sm55[0])[1]):
        tt = np.float16(np.asarray(range(1, 5)).tolist())/10
        tt1 = sm55[i][:, j].tolist()
        ttt = np.asarray(sq55)
        mm5[i, j] = np.average(tt1)
        mq5[i, j] = ttt[i, j]



pr = model.predict(datx)
pr1 = np.zeros(len(pr))

########### ONE-HOT REPRESENTATION OF TEST OUTPUT ##########
prt = np.zeros((np.shape(pr)[0], np.shape(pr)[1]))
aa1 = np.max(pr, 1)
for i in range(len(aa1)):
    for j in range(10):
        if pr[i, j] == aa1[i]:
            prt[i, j] = 1
        else:
            prt[i, j] = 0

for i in range(len(x_test)):
    for j in range(10):
        if int(prt[i, j]) == 1:
            pr1[i] = j


########### FIND TEST SAMPLE LOCATIONS WHERE MODEL MADE WRONG PREDICTIONS ##########
cre = np.zeros(len(pr))
cre1 = np.zeros(len(pr))
ct = 0
for i in range(len(pr)):
    if int(pr1[i]) == yts[i]:
        cre[i] = 1
        cre1[i] = 0

        ct += 1
    else:
        cre[i] = 0
        cre1[i] = 1

cre = cre.astype(int)
cre1 = cre1.astype(int)


########## CORRELATE UNCERTAINTY MEASURES WITH WRONG PREDICTIONS TO EVALUATE UQ QUALITY ###########
npp = [0, 1, 0]

sm5 = np.sum(mm5 * npp, 1)

sm5b = np.sum(mq5 * npp, 1)



auc5, auc8, f15, f18, fs5, \
fs8, cv5, cv8, prs5, prs8, sprs5, sprs8, pb5, pb8, f1s5, f1s8, puc5, puc8, md5, md8 = perf.roc(model, datx, daty[0], sm5, sm8)
