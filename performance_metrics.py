import numpy as np


class perf:

    def roc(model, x_test, yts, sm5, sm8):

        pmt = model.predict(x_test)

        prt = np.zeros((len(pmt), 10))
        pr1 = np.zeros(len(pmt))

        aa1 = np.max(pmt, 1)
        for i in range(len(aa1)):
            for j in range(10):
                if pmt[i, j] == aa1[i]:
                    prt[i, j] = 1
                else:
                    prt[i, j] = 0

        for i in range(len(x_test)):
            for j in range(10):
                if int(prt[i, j]) == 1:
                    pr1[i] = j

        cre = np.zeros(len(pmt))
        cre1 = np.zeros(len(pmt))
        ct = 0
        for i in range(len(pmt)):
            if int(pr1[i]) == yts[i]:
                cre[i] = 1
                cre1[i] = 0

                ct += 1
            else:
                cre[i] = 0
                cre1[i] = 1

        from sklearn import metrics
        import matplotlib.pyplot as plt
        from numpy import cov
        from scipy.stats import pearsonr as per
        from scipy.stats import spearmanr as sper
        from scipy.stats import pointbiserialr as pbs

        L = cre1
        L = L.astype(bool)


        fpr5, tpr5, _ = metrics.roc_curve(cre1, sm5)
        auc5 = metrics.roc_auc_score(cre1, sm5)
        f15 = metrics.precision_recall_curve(cre1, sm5)
        f15 = [f15[0][0:-1], f15[1][0:-1], f15[2][0:-1]]
        puc5 = metrics.auc(f15[1], f15[0])
        fs5 = metrics.average_precision_score(cre1, sm5)
        cv5 = cov(cre1, sm5)[0][1]
        prs5, _ = per(cre1, sm5)
        sprs5, _ = sper(cre1, sm5)
        pb5, _ = pbs(cre1, (sm5 - np.min(sm5))/(np.max(sm5) - np.min(sm5)))
        sa5 = (sm5 - np.min(sm5))/(np.max(sm5) - np.min(sm5))
        # cv5 = cov(cre1, sa5)[0][1]

        h51, h52 = sm5[L], sm5[~L]
        hn = np.concatenate([h51, h52])
        hn = (hn - np.min(hn))/(np.max(hn) - np.min(hn))
        h51 = hn[0:len(h51)]
        h52 = hn[len(h51): len(h52)]
        md5 = abs(np.mean(h51) - np.mean(h52))

        sd = f15
        f1s5 = 0
        for u in range(len(sd[0])):
            f1s5 += 2*sd[0][u]*sd[1][u]/(sd[0][u] + sd[1][u])
        f1s5 /= len(sd[0])


        fpr8, tpr8, _ = metrics.roc_curve(cre1, sm8)
        auc8 = metrics.roc_auc_score(cre1, sm8)
        f18 = metrics.precision_recall_curve(cre1, sm8)
        f18 = [f18[0][0:-1], f18[1][0:-1], f18[2][0:-1]]
        puc8 = metrics.auc(f18[1], f18[0])
        fs8 = metrics.average_precision_score(cre1, sm8)
        cv8 = cov(cre1, sm8)[0][1]
        prs8, _ = per(cre1, sm8)
        sprs8, _ = sper(cre1, sm8)
        pb8, _ = pbs(cre1, (sm8 - np.min(sm8))/(np.max(sm8) - np.min(sm8)))
        sa8 = (sm8 - np.min(sm8))/(np.max(sm8) - np.min(sm8))
        # cv8 = cov(cre1, sa8)[0][1]

        h81, h82 = sm8[L], sm8[~L]
        hn = np.concatenate([h81, h82])
        hn = (hn - np.min(hn))/(np.max(hn) - np.min(hn))
        h81 = hn[0:len(h81)]
        h82 = hn[len(h81): len(h82)]
        md8 = abs(np.mean(h81) - np.mean(h82))

        sd = f18
        f1s8 = 0
        for u in range(len(sd[0])):
            f1s8 += 2*sd[0][u]*sd[1][u]/(sd[0][u] + sd[1][u])
        f1s8 /= len(sd[0])

        lw = 2

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.hist(h51, density=True, alpha = 0.5, color='r')
        plt.hist(h52, density=True, alpha = 0.5, color='b')
        # plt.grid(linestyle='dotted')
        plt.yticks([])
        plt.xlabel("UNCERTAINTY ESTIMATES (NORMALIZED): BLUE = CORRECT PREDICTIONS, PINK = WRONG PREDICTIONS")
        plt.title('QIPF')

        plt.subplot(1, 2, 2)
        plt.hist(h81, density=True, alpha = 0.5, color='r')
        plt.hist(h82, density=True, alpha = 0.5, color='b')
        # plt.grid(linestyle='dotted')
        plt.yticks([])
        plt.title('ENSEMBLE')
        # plt.xlabel('NOISE SEVERITY')
        # plt.ylabel('ERROR CLASS MEAN DIFF')

        plt.figure()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.plot(fpr5, tpr5, linestyle='-', label="QIPF (AUC = %0.2f)" % auc5)
        plt.plot(fpr8, tpr8, linestyle='--', label="ENSEMBLE (AUC = %0.2f)" % auc8)
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.legend(loc=4)
        plt.grid(linestyle='dotted')
        plt.xlabel('FALSE POSITIVE RATE')
        plt.ylabel('TRUE POSITIVE RATE')
        plt.show()

        plt.figure()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.plot(f15[1], f15[0], linestyle='-', label="QIPF (AUC = %0.2f)" % puc5)
        plt.plot(f18[1], f18[0], linestyle='--', label="ENSEMBLE (AUC = %0.2f)" % puc8)
        # plt.plot([0, 1], [1, 0], color='black', lw=lw, linestyle='--')
        plt.legend(loc=4)
        plt.grid(linestyle='dotted')
        plt.ylabel('PRECISION')
        plt.xlabel('RECALL')
        plt.show()

        return auc5, auc8, f15, f18, \
               fs5, fs8, cv5, cv8, prs5, \
               prs8, sprs5, sprs8, pb5, \
               pb8, f1s5, f1s8, puc5, \
               puc8, md5, md8

