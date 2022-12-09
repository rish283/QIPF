############# IMPORT NECESSARY LIBRARIES ###########################################
import numpy as np
import matplotlib.pyplot as plt
from colour import Color
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy, mean_squared_error
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Softmax, ReLU
#####################################################################################

MediumGray = Color("Gray")
purple = Color("purple")
green = Color("green")
yellow = Color("yellow")
red = Color("red")

################################ GENERATE DATA FOR REGRESSION #############################

########### functions to generate regression data (input = x, desired = function output)) ##################
############### function 1 #############
def f(x, alpha=4., beta=13.):
    w = np.random.normal(0, 0.03)
    return x + np.sin(alpha*(x+w)) + np.sin(beta*(x+w))


# ############### function 2 #############
# def f1(x, alpha=4., beta=13.):
#     w = np.random.normal(0, 0.3)
#     return x*np.sin(x+w)

############### function to get k random samples in range (a, b) ##############
def sample(a, b, k):
    assert b>a
    return np.random.random(k)*(b-a) + a

############### generate x and y (i.e. f(x)) for regression, but only in specific regions: (-1.2, 0.1) and (0.7, 1) ##################
x_list = np.r_[sample(-1.2, 0.1, 80), sample(0.7, 1., 30)]
# x_list = np.r_[sample(-5, 5, 60)]
X_train = np.asarray(sorted(x_list)).reshape(-1, 1)
y_train = np.asarray([f(x) for x in X_train]).reshape(-1, 1)

########################### DEFINE MODEL CLASS (SIMPLE OVERPARAMETERIZED MLP) ##########################################
class mllp(Sequential):
    def __init__(self, input_shape, nb_classes):
        super().__init__()

        self.add(Dense(100, activation='relu', input_shape=input_shape))#, kernel_regularizer= regularizers.l1(0.01)))
        self.add(Dense(100, activation='relu'))#, kernel_regularizer= regularizers.l1(0.01)))
        self.add(Dense(100, activation='relu'))#, kernel_regularizer= regularizers.l1(0.01)))
        self.add(Dense(nb_classes))
        # self.add(ReLU())

        self.compile(optimizer='adam',
                    loss=mean_squared_error)
        return

############################### BUILD MODEL #################
model = mllp(X_train[0].shape, 1)

####################### TRAIN THE MODEL WITH REGRESSION DATA ##################
model.fit(X_train, y_train, epochs=5000, batch_size=len(X_train))

################### GENERATE TEST DATA (FROM THE ENTIRE DATA REGION) ##############
x_test = np.linspace(-2.5, 2, 150).reshape(-1, 1)
y_test = np.asarray([f(x) for x in x_test]).reshape(-1, 1)
# x_test = np.linspace(-10, 10, 100).reshape(-1, 1)

######################## MODEL OUTPUT FOR TEST DATA ###########################
pr = model.predict(x_test)


######################### QIPF PARAMETERS ################################
orr = 14                   #### ORDER = orr/ttt -2
bwf = 200                  #### Silverman kernel bandwidth multiplier
ttt = 2                    #### ttt = 2 (ever order moments)

######################## IMPLEMENT QIPF #############################
import math
import skimage.measure as sk
from statsmodels.nonparametric.bandwidths import bw_silverman as bw

################## EXTRACT MODEL LAYERS IN fc #######################
fc = []
for i in range(len(model.layers)):
    fc.append(model.layers[i])

################## EXTRACT LAYER WEIGHTS ############################
hm1 = sk.block_reduce(fc[0].get_weights()[0], (1, 1), np.mean).flatten()
hm2 = sk.block_reduce(fc[1].get_weights()[0], (1, 1), np.mean).flatten()
hm3 = sk.block_reduce(fc[2].get_weights()[0], (1, 1), np.mean).flatten()
hm4 = sk.block_reduce(fc[3].get_weights()[0], (1, 1), np.mean).flatten()
# hm5 = sk.block_reduce(fc[7].get_weights()[0], (1, 5), np.mean).flatten()

#################### CONCATENATE AND NORMALIZE WEIGHTS ####################
hmt = np.concatenate((hm1, hm2, hm3, hm4))
hmt = hmt.flatten()
hmt = (hmt - np.mean(hmt)) / np.std(hmt)

############################ OBTAIN LAST LAYER OUTPUT OF MODEL (FOR TEST INPUTS) = pmt ##############
x = fc[-2].output
preds = fc[-1](x)
modelq = Model(inputs=model.input, outputs=preds)

xtest = x_test

pmt = modelq.predict(xtest)

########################## NORMALIZE WRT TO TRAIN OUTPUT ###########################
rmt = modelq.predict(X_train)
pmt1 = np.max(pmt, 1)
rmt1 = np.max(rmt, 1)
pmtn = (pmt1 - np.mean(rmt1)) / np.std(rmt1)



######################## START QIPF #################
es = 0.2
n = len(xtest)
ct = 0
print('start')

s0m = []
sqp = []
sigg1 = bwf * np.average(bw(hmt))

w1 = np.zeros((len(xtest), 3))
N1 = len(xtest)
N2 = len(hmt)
import time

start = time.time()

for i in range(len(xtest)):


    jh = pmtn[i]
    jh = [jh - es, jh, jh + es]

    ct += 1
    if ct % 10 == 0:
        print(" iter #: ", ct, ' / ', N2)

    w1[i, 0] = (1 / N2) * np.sum(np.exp(-(np.power(jh[0] - hmt, 2)) / (2 * sigg1 ** 2)))
    w1[i, 1] = (1 / N2) * np.sum(np.exp(-(np.power(jh[1] - hmt, 2)) / (2 * sigg1 ** 2)))
    w1[i, 2] = (1 / N2) * np.sum(np.exp(-(np.power(jh[2] - hmt, 2)) / (2 * sigg1 ** 2)))

    jh = [[w1[i, 0]], [w1[i, 1]], [w1[i, 2]]]
    w0 = np.sqrt(jh)
    x = w0
    n = np.arange(1, orr + 1)
    fn = np.floor(n / 2)
    p = np.arange(0, orr + 1)
    x = 2 * x
    lex = len(x)
    lenn = len(n)

    if p[0] == 0:
        xp = np.power(x, p[1::])
        xp = np.concatenate([np.ones((lex, 1)), xp], axis=1)
    else:
        xp = np.power(x, p)

    H = np.zeros((lex, lenn))
    H = np.float64(H)
    yy = np.zeros(lenn)
    yy = np.float64(yy)

    for k in range(lenn):
        for m in range(int(fn[k]) + 1):
            is_the_power = p == n[k] - (2 * m)
            jj = (1 - 2 * np.mod(m, 2)) / math.factorial(m) / math.factorial(n[k] - (2 * m)) * xp[:,
                                                                                               is_the_power]
            H[:, k] += jj[:, 0]

        ll = math.factorial(n[k])
        H[:, k] = ll * H[:, k]
    wy = H
    sg = sigg1 ** 2
    qe = np.gradient(np.gradient(np.abs(wy), axis=1), axis=1)
    qe1 = np.abs(wy)
    vc = np.multiply((sg / 2), np.divide(qe, qe1))
    r = np.zeros((np.shape(vc)[0], int((np.shape(vc)[1] / ttt)) - 1))
    for qk in range(1, int(orr / ttt)):
        if len(wy) == 1:
            r[:, qk - 1] = np.abs(vc[:, (ttt * qk) - 1])
        else:
            r[:, qk - 1] = vc[:, (ttt * qk) - 1] - np.min(vc[:, (ttt * qk) - 1])
    r = r.T
    qn0 = np.double(r[0:-1])
    sk = qn0
    qp = np.double(r[-1])

    s0m.append(sk)
    sqp.append(qp)

stop = time.time()
duration = stop - start
print(duration)


####################### QIPF UNCERTAINTY MODES #########################################
sm = s0m


######################### PROCESS AND NORMALIZE THE UNCERTAINTY MODES ##################
mm = np.zeros((len(sm), int(orr/ttt) - 2))
for i in range(len(sm)):
    mm[i, :] = sm[i][:, 1]

mmq = mm[:, 0:-1]

# mmq = (mmq - np.min(mmq))/(np.max(mmq) - np.min(mmq))
mm2 = np.copy(mmq)

for i in range(len(mmq[0])):
    mm2[:, i] = mmq[:, i] - np.mean(mmq, axis=1)


############################## NORMALIZE UNCERTAINTY MODES AROUND MODEL PREDICTIONS ####################
qq = mm2 + pr
a0 = x_test.ravel()

a1 = pr.ravel() + np.max(qq, axis=1)
a2 = pr.ravel() - np.max(qq, axis=1)

a1m = pr.ravel() + np.mean(mmq, axis=1)
a2m = pr.ravel() - np.mean(mmq, axis=1)

a1s = pr.ravel() + np.std(qq, axis=1)
a2s = pr.ravel() - np.std(qq, axis=1)
##########################################################################################################

################################## PLOT TRUE FUNCTION, MODEL PREDICTION, QIPF UNCERTAINTY ENVELOPES ####################
################################## PINK BANDS: UNSEEN REGIONS BY MODEL, WHITE BANDS: SEEN REGIONS BY MODEL #############
plt.figure()
# plt.scatter(X_train, y_train, alpha=0.2, marker='o')
plt.plot(x_test, y_test, '--', color='m', lw=2, label='True Function')
plt.plot(x_test, pr, '--', color='b', lw = 2, alpha = 1, label='Model Prediction')
# plt.plot(x_test, np.mean(mmq, axis=1), '--')
# plt.plot(x_test, np.zeros(len(x_test)), '--', color='k')
for i in range(np.shape(qq)[1]):
    plt.plot(x_test, qq[:, i], lw=1.2, label='Mode: ' + str(i))

plt.fill_between(a0, a1s, a2s, color='b', alpha=0.18)
plt.axvspan(0.1, 0.7, color='red', alpha=0.1)
plt.axvspan(-2.5, -1.2, color='red', alpha=0.1)
plt.axvspan(1, 2, color='red', alpha=0.1)
# plt.xlim([-2.5, 2])
# plt.ylim([-6, 7])
plt.legend(loc = 'upper left')


# plt.figure()
# plt.scatter(X_train, y_train, alpha=0.4, marker='o')
# plt.plot(x_test, pr, '--', color='b', lw = 2, alpha = 1)
# # plt.plot(x_test, np.mean(mmq, axis=1), '--')
# # plt.plot(x_test, np.zeros(len(x_test)), '--', color='k')
# plt.plot(x_test, qq)
# #
# plt.fill_between(a0, a1m, a2m, color='b', alpha=0.2)
# plt.axvspan(-10, -5, color='red', alpha=0.1)
# plt.axvspan(5, 10, color='red', alpha=0.1)
# # plt.axvspan(1, 2, color='red', alpha=0.1)
# plt.xlim([-9, 9])
# plt.ylim([-10, 10])
