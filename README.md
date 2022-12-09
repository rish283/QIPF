# QIPF
Implementation of the QIPF framework for model uncertainty quantification. Link to related papers: 
- https://arxiv.org/abs/2109.10888
- https://arxiv.org/abs/2001.11495
- https://arxiv.org/abs/2211.01999


Implementations contains ROC-AUC and PR-AUC metric comparison between QIPF and Ensemble methods. We chose ensemble methods for the code demonstration since it currently achieves state-of-the-art performance in many applications.
=====================================================
Files:
main_UQ.py: main file containing initializations and call functions for
UQ methods.
UQ_methods.py: contains QIPF and ensemble functions
performance_metrics.py: contains function to evaluate ROC-AUC, PR-AUC
REGRESSION EXAMPLE QIPF.py: Contains toy regression problem and QIPF implementation for fundamental understanding
sine_test.py: Contains code for QIPF decomposition of sine-wave signal.
=====================================================
Folders:
Data folder: Contains K-MNIST data
Models folder: Contains LeNet initializer function (lenet.py), trained LeneT model on KMNIST (md_kmni.h5) and ensemble/KMNIST folder contains models trained on KMNIST with different initializations for the ensemble method.
