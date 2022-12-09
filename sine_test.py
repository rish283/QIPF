import numpy as np
import math
from statsmodels.nonparametric.bandwidths import bw_silverman as bw
import matplotlib.pyplot as plt

###################### GENERATE SINE WAVE SIGNAL #################################
# t = np.arange(0, 100, 0.1)
# A1 = np.sin(t)
# A = (A1 - np.min(A1, axis=0))/(np.max(A1, axis=0) - np.min(A1, axis=0))
# A = A - np.mean(A)

T = 1000
f = 10
x = np.arange(0, T)
A = np.sin(2*f*np.pi*x/T) #+np.cos(8*np.pi*x/T)

plt.plot(x, A)
plt.title('Sine wave')
plt.xlabel('Time')

###################### QIPF PARAMETERS ################################
orr = 14    # NO. OF MODES = (orr/2)-2
ttt = 2     # GAPS BETWEEN MODES (EX: IT TTT=2, EXTRACT MODES 0, 2, 4 ...). We extract even orders
sigg1 = 0.5 # KERNEL WIDTH

#################### DATA SPACE FOR QIPF ANALYSIS #############################
t1 = np.arange(-3, 3, 0.01)
N1 = len(t1)

################### INITIALIZE QIPF VARIABLES #########################
smv1 = np.zeros((N1, int((orr / 2) - 2)))
w1 = np.zeros((N1, 3))

################## NEIGHBORHOOD SIZE FOR GRADIENT (LAPLACIAN) COMPUTATION: TAKEN AS THE DATA STD. DEV ###############
es = 2*np.std(A, axis=0)


sigr = []
ct = 0


######################################## QIPF IMPLEMENTATION START ###############################################
for i1 in range(N1):
    ct += 1
    print("1_iter: ", ct)

    #### EVALUATE WAVE-FUNCTION AT A SAMPLE LOCATION AND NEIGHBORING POINTS ###################
    w1[i1, 0] = (1 / N1) * np.sum(np.exp(-(np.power(t1[i1] - A, 2)) / (2 * sigg1 ** 2)))
    w1[i1, 1] = (1 / N1) * np.sum(np.exp(-(np.power(t1[i1] - es - A, 2)) / (2 * sigg1 ** 2)))
    w1[i1, 2] = (1 / N1) * np.sum(np.exp(-(np.power(t1[i1] + es - A, 2)) / (2 * sigg1 ** 2)))

    s0m = []
    jh = [[w1[i1, 1]], [w1[i1, 0]], [w1[i1, 2]]]

    w0 = np.sqrt(jh)
    x = w0

    ##################### HERMITE DECOMPOSITION OF THE WAVE - FUNCTION #########################
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

            H[:, k] += jj[:, 0] ###### HERMITE POLYNOMIAL PROJECTIONS

        ##### NORMALIZATION OF HERMITE PROJECTIONS ########
        # ll = math.factorial(n[k])
        # H[:, k] = ll * H[:, k]


    wy = H
    sg = sigg1 ** 2
    qe = np.gradient(np.gradient(np.abs(wy), axis=1), axis=1) # COMPUTE LAPLACIAN
    qe1 = np.abs(wy)
    vc = np.multiply((sg / 2), np.divide(qe, qe1))
    r = np.zeros((np.shape(vc)[0], int((np.shape(vc)[1] / ttt)) - 1))

    for qk in range(1, int(orr / ttt)):
        if len(wy) == 1:
            r[:, qk - 1] = np.abs(vc[:, (ttt * qk) - 1])
        else:
            r[:, qk - 1] = vc[:, (ttt * qk) - 1] - np.min(vc[:, (ttt * qk) - 1]) ##### QIPF CORRESPONDING TO EACH MODE

    r = r.T
    qn0 = np.double(r[0:-1])
    sk = qn0

    smv1[i1, :] = sk[:, 1]


###### NORMALIZE MODES AND WAVE_FUNCT FOR EASIER VISUALIZATION #######
smvn = (smv1 - np.min(smv1))/(np.max(smv1) - np.min(smv1))
wn = (w1 - np.min(w1))/(np.max(w1) - np.min(w1))


############# PLOT WAVE-FUNCTION/KME (w1) AND MODES (smv1) IN THE DATA SPACE
c = ['r', 'g', 'b', 'm', 'k', 'y']
md = ['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4', 'Mode 5', 'Mode 6']
plt.figure()
plt.plot(w1[:, 0], '--', label = 'Wave-func/KME/PDF')
for i in range(np.int(orr/2) -2):
    plt.plot(smv1[:, i], color = c[i], label = md[i])
plt.legend(loc = 1)
