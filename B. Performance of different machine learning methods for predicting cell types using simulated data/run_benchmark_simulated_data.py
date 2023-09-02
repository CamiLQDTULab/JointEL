# Run: python approach_id data_set 
# approach_id: 1: JointEL, 2: ZScale, 3: RangeScale
# e.g: python 1 GeqN1k

import numpy as np
from numpy import genfromtxt
from sklearn.decomposition import PCA
from Jvis import JUMAP, UMAP
import sys
from timeit import default_timer as timer
import time
from sklearn import preprocessing

data_set = sys.argv[2] #sys.argv[1] #'GeqN1kD1k'
data_name = "output_jumap/" + data_set #output

def shuffle_rows(expression_matrix, prop, random_seed):
    np.random.seed(random_seed)
    n_rows, n_cols = expression_matrix.shape
    n_elem = round(prop*n_cols)
    s = np.arange(n_cols)
    row_id = list(np.random.choice(s, size=n_elem, replace=False))
    v = expression_matrix[:, row_id]
    np.random.shuffle(np.transpose(v))
    expression_matrix[:, row_id] = v
    
def normalize_matrix(A, option = "Frobenius"):
    if option == "Frobenius":
        return (A/np.linalg.norm(A))
    elif option == "max":
        return (A/A.max())
    else:
        "Not implemented yet"

# Read the expr matrix
expr_mat = genfromtxt('../sim_splat/data/'+data_set+'_rna.csv',  delimiter=',')
labels_true = genfromtxt('../sim_splat/data/'+data_set+'_labels.csv', delimiter=',')
# labels_true = labels_true.values.flatten()
expr_mat_log_t = expr_mat
print("RNA_shape", expr_mat.shape)
expr_reduced = PCA(n_components=50).fit_transform(expr_mat_log_t)

# Read ADT
adt_mat = genfromtxt('../sim_splat/data/'+data_set+'_adt.csv', delimiter=',')
print("ADT shape: ", adt_mat.shape)

start = time.time()
noise_level = 0.2
# for noise_level in [0.2]:
# Create noise
print("Dataset: ", data_set)
print("Noise level: ", noise_level)
expr_mat_shuffle = np.copy(expr_mat)
shuffle_rows(expr_mat_shuffle.T, prop = noise_level, random_seed=0)

if expr_mat_shuffle.shape[1] <500:
    expr_reduced_shufle = expr_mat_shuffle
else:
    expr_reduced_shufle = PCA(n_components=50).fit_transform(expr_mat_shuffle)

maxIter = 25 if adt_mat.shape[0] < 4000 else 15

joint_umap_obj = JUMAP(init='random')
data = {'modal-1': expr_reduced, 'modal-2': adt_mat,  'modal-noise': expr_reduced_shufle}
# joint_umap = joint_umap_obj.fit_transform(X = data, method = 'auto', ld = 0.5, max_iter=maxIter)
joint_umap = joint_umap_obj.fit(data, alpha=[0.45, 0.45, 0.1])


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

def run_ML(X, y, approach="Default"):
    score = []
    methods = []
    n_loops = 10
    n_samples = y.shape[0]
    for i in range(n_loops):
        path_dir = '/data/hoan/Jvis/multiomics_development/proof_of_principle/output_classification/' + data_set + '_run_'+str(i)+'_'+approach
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
        methods.append('SVM')
        print(methods[-1], end =', ')
        clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        np.savetxt(path_dir + "_SVM_labels.csv", y_predict, delimiter=",")
        score.append(f1_score(y_predict, y_test, average='macro'))
        # Decision Tree
        methods.append('Decision Tree')
        print(methods[-1], end =', ')
        clf = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
        np.savetxt(path_dir + "_DecisionTree_labels.csv", y_predict, delimiter=",")
        y_predict = clf.predict(X_test)
        score.append(f1_score(y_predict, y_test, average='macro'))
        # RF
        methods.append('RF')
        print(methods[-1], end =', ')
        clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        np.savetxt(path_dir + "_RandomForest_labels.csv", y_predict, delimiter=",")
        score.append(f1_score(y_predict, y_test, average='macro'))
        # Neural network
        methods.append('Neural network')
        print(methods[-1], end =', ')
        clf = MLPClassifier(alpha=1, max_iter=2000).fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        np.savetxt(path_dir + "_NeuralNet_labels.csv", y_predict, delimiter=",")
        score.append(f1_score(y_predict, y_test, average='macro'))
        # Adaboost
        methods.append('Adaboost')
        print(methods[-1], end =', ')
        clf = AdaBoostClassifier().fit(X_train, y_train)
        np.savetxt(path_dir + "_Adaboost_labels.csv", y_predict, delimiter=",")
        y_predict = clf.predict(X_test)
        score.append(f1_score(y_predict, y_test, average='macro'))
        # GradientBoostingClassifier
        methods.append('Gradient Boost Decision Tree')
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0).fit(X_train, y_train)
        print(methods[-1], end ='\n')
        np.savetxt(path_dir + "_GBDT_labels.csv", y_predict, delimiter=",")
        y_predict = clf.predict(X_test)
        score.append(f1_score(y_predict, y_test, average='macro'))
        ## K-NN 
        methods.append('kNN')
        clf = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
        print(methods[-1], end ='\n')
        np.savetxt(path_dir + "_NearestNeighbors_labels.csv", y_predict, delimiter=",")
        y_predict = clf.predict(X_test)
        score.append(f1_score(y_predict, y_test, average='macro'))
        # Naive Bayes
        methods.append('NaiveBayes')
        clf = GaussianNB().fit(X_train, y_train)
        print(methods[-1], end ='\n')
        np.savetxt(path_dir + "_NaiveBayes_labels.csv", y_predict, delimiter=",")
        y_predict = clf.predict(X_test)
        score.append(f1_score(y_predict, y_test, average='macro'))
        
    
approach = int(sys.argv[1])
# Embedding
if approach == 1:
    print("Joint embedding")
    joint_embedding = joint_umap_obj.embedding_
    run_ML(joint_embedding, labels_true, 'JointEL')

# z-score
if approach == 2:
    print("z-transform")
    concat = np.concatenate((expr_mat_log_t, adt_mat, expr_mat_shuffle), axis = 1)
    X_train = concat
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    print("X_scaled", X_scaled.shape)
    run_ML(X_scaled, labels_true, 'ZScale')

# 0, 1 standard
if approach == 3:
    print("0-1 scale")
    print("Cross and validation over alpha")
    concat = np.concatenate((normalize_matrix(expr_mat_log_t, "max"), normalize_matrix(adt_mat, "max"), normalize_matrix(expr_mat_shuffle, "max")), axis = 1)
    print('concat', concat.shape)
    run_ML(concat, labels_true, 'RangeScale')
    
