# Author: Van Hoan Do <vanhoan310@gmail.com>
# License: BSD 3 clause (C) 2020
from numpy import genfromtxt
import numpy as np 
import matplotlib
import pylab as plt
import pandas as pd
from Jvis import UMAP, JUMAP, JTSNE
import time


# In[3]:


## Run ML methods    
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import datasets
from sklearn import svm
import random
# X, y = datasets.load_iris(return_X_y=True)


# In[4]:


def run_ML(X, y, approach="Default"):
    score = []
    methods = []
    n_loops = 1
    n_samples = y.shape[0]
    for i in range(n_loops):
        path_dir = 'output/' + data_set + '_run_'+str(i)+'_'+approach
        print('Run: ', i)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        # print(X_train.shape, X_test.shape)
        random.seed(i)
        print("n_samples: ", n_samples)
        train_idx = random.sample(range(n_samples), int(n_samples*0.8))
        test_idx = [i for i in range(n_samples) if i not in train_idx]
        print(train_idx[:10])
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        # Save the test true labels
        np.savetxt(path_dir + "_test_true_labels.csv", y_test, delimiter=",")
        print(X_train.shape, X_test.shape)
        
        # SVM
        start_time = time.time()
        methods.append('SVM')
        print(methods[-1], end =', ')
        clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        print("--- %s seconds ---" % (time.time() - start_time))
        np.savetxt(path_dir + "_SVM_labels.csv", y_predict, delimiter=",")
        score.append(f1_score(y_predict, y_test, average='macro'))
        # Decision Tree
        start_time = time.time()
        methods.append('Decision Tree')
        print(methods[-1], end =', ')
        clf = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
        np.savetxt(path_dir + "_DecisionTree_labels.csv", y_predict, delimiter=",")
        y_predict = clf.predict(X_test)
        print("--- %s seconds ---" % (time.time() - start_time))
        score.append(f1_score(y_predict, y_test, average='macro'))
        # Random Forest
        start_time = time.time()
        methods.append('RF')
        print(methods[-1], end =', ')
        clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        print("--- %s seconds ---" % (time.time() - start_time))
        np.savetxt(path_dir + "_RandomForest_labels.csv", y_predict, delimiter=",")
        score.append(f1_score(y_predict, y_test, average='macro'))
        # Neural network
        start_time = time.time()
        methods.append('Neural network')
        print(methods[-1], end =', ')
        clf = MLPClassifier(alpha=1, max_iter=2000).fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        print("--- %s seconds ---" % (time.time() - start_time))
        np.savetxt(path_dir + "_NeuralNet_labels.csv", y_predict, delimiter=",")
        score.append(f1_score(y_predict, y_test, average='macro'))
        # Adaboost
        start_time = time.time()
        methods.append('Adaboost')
        print(methods[-1], end =', ')
        clf = AdaBoostClassifier().fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        print("--- %s seconds ---" % (time.time() - start_time))
        np.savetxt(path_dir + "_Adaboost_labels.csv", y_predict, delimiter=",")
        score.append(f1_score(y_predict, y_test, average='macro'))
        # Naive Bayes
        start_time = time.time()
        methods.append('NaiveBayes')
        clf = GaussianNB().fit(X_train, y_train)
        print(methods[-1], end ='\n')
        y_predict = clf.predict(X_test)
        print("--- %s seconds ---" % (time.time() - start_time))
        np.savetxt(path_dir + "_NaiveBayes_labels.csv", y_predict, delimiter=",")
        score.append(f1_score(y_predict, y_test, average='macro'))
        
    # Print statistics
    n_methods = 6
    score_np = np.array(score)
    # Each column is a method
    print(methods[:n_methods])
    print(np.mean(score_np.reshape((n_loops, n_methods)), axis=0))
    

def normalize_matrix(A, option = "Frobenius"):
    if option == "Frobenius":
        return (A/np.linalg.norm(A))
    elif option == "max":
        return (A/A.max())
    else:
        "Not implemented yet"

rna_matrix = np.genfromtxt('../data/cbmc_rna_pca.csv', delimiter=',')
adt_matrix = np.genfromtxt('../data/cbmc_adt.csv', delimiter=',')

y = np.genfromtxt('../data/cbmc_citefuse_labels.csv', delimiter=',')
cell_types = ["CD4+ T", "CD14+ Mono", "B", "NK", "CD34+", "pDCs", "CD8+ T", "CD16+ Mono.", "Eryth", "DC"] 
labels_true = y.astype(int)

data_set = 'CBMC'
metrics = 'correlation'; n_NN = 15;min_dist = 0.04
start_time = time.time()
joint_umap_obj = JUMAP(init='random', metric=metrics, min_dist=min_dist)
data = {'rna': rna_matrix, 'adt': adt_matrix}
joint_umap = joint_umap_obj.fit(data, alpha=[0.5, 0.5])
joint_embedding = joint_umap_obj.embedding_
JUMAP_time = time.time() - start_time
print(JUMAP_time)

# approach = int(sys.argv[1])
# Embedding
# if approach == 1:
print("Joint embedding")
# joint_embedding = Z
run_ML(joint_embedding, labels_true, 'JointEL')

# z-score
# if approach == 2:
print("z-transform")
concat = np.concatenate((rna_matrix, adt_matrix), axis = 1)
X_train = concat
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)
print("X_scaled")
run_ML(X_scaled, labels_true, 'ZScale')

# 0, 1 standard
# if approach == 3:
print("0-1 scale")
concat = np.concatenate((normalize_matrix(rna_matrix, "max"), normalize_matrix(adt_matrix, "max")), axis = 1)
print('concat')
run_ML(concat, labels_true, 'RangeScale')



