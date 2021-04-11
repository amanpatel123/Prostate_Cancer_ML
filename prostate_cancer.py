import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn import svm                                  

from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import IsolationForest

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from statistics import mean
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

MODEL = svm.SVC(kernel='linear')
def load_genomic_data(filename):
     # load the geneomic data
    genomic_df = pd.read_csv(filename)

    # Setting index to first column, else it will add its own indexing while doing transpose
    genomic_df.set_index('ID', inplace = True)
    
    # Need to take transpose since I want genes/features to be columns and each row should represent a patient information
    genomic_df = genomic_df.T
    
    # removing features with only zero values for all patients
    return genomic_df.loc[:, (genomic_df != 0).any(axis = 0)]    
    
def read_data(gfilename, cfilename):
    # Feature set, load geonomic data
    X = load_genomic_data(gfilename)
    
    # load the clinical data
    clinical_df = pd.read_csv(cfilename)
    print("Shape of genomic data: ", X.shape, " and Shape of clinical data: ", clinical_df.shape, "thus looks like we donot have genetic data for 5 patients, hence removing them")
    clinical_df = clinical_df.drop(labels=[213,227,297,371,469], axis=0)
    print("After droping 5 patients whose data were missing:\nShape of genomic data: ", X.shape, " and Shape of clinical data: ", clinical_df.shape, "\n")
    
    print("-- Checking if all patient ID's in genetic data set and clinical dataset matches\n")
    if(X.index.all() == clinical_df['PATIENT_ID'].all()):
        print("-- Yes, patient ID's in genetic data set and clinical dataset matches\n")
    else:
        print("Nope, patient ID's in genetic data set and clinical dataset do not match")
    
    y =  clinical_df['GLEASON_SCORE']                   
    
    return X, y


def visualize_data(X, y, title):
    # Visualizing dataset for outliers, using PCA prioir to LDA to prevent overfitting (https://stats.stackexchange.com/q/109810)
    pca = PCA(n_components=10)
    pca_reduced_data = pca.fit_transform(X,y)
    
    lda = LinearDiscriminantAnalysis(n_components = 2)
    pca_lda_reduced_data = lda.fit_transform(pca_reduced_data, y)
    
    # NOTE: Gleason score ranges from 6-10
    label = [6, 7, 8, 9, 10]
    colors = ['red','green','blue','purple','pink']
    
    fig = plt.figure(figsize=(6,6))
    plt.scatter(pca_lda_reduced_data[:,0], pca_lda_reduced_data[:,1], c=y, cmap=matplotlib.colors.ListedColormap(colors), alpha=0.7)
    plt.title(title)
    
    
def prepare_inputs(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)             
    return X_train_norm, X_test_norm

def remove_low_variance_feature(X):
    sel = VarianceThreshold()
    return sel.fit_transform(X)
    
    
def filter_feature_selection(X_train_norm, y_train, X_test_norm, score_function):
    best_k = SelectKBest(score_func=score_function, k=300)
    fit = best_k.fit(X_train_norm, y_train)
    X_train_fs = fit.transform(X_train_norm)
    X_test_fs = fit.transform(X_test_norm)
    
    # DONOT REMOVE 
    # dfscores = pd.DataFrame(fit.scores_)
    # dfcolumns = pd.DataFrame(X_train.columns)
    # featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    # featureScores.columns = ['Features','Score']
    # best = featureScores.nlargest(40,'Score')
    # for f in best['Features']:
    #     featureVoting[f] = featureVoting.get(f, 0) + 1 
    #best.plot(x='Features', y="Score", kind="bar")
    return X_train_fs, X_test_fs

def forward_feature_selecion(X_train_fs, y_train, X_test_fs):
    sfs = SFS(MODEL, 
		k_features=13,
		forward=True,
		floating = False,
		scoring = 'accuracy',
		cv = 5)
    fit = sfs.fit(X_train_fs, y_train)
    X_train_wfs = fit.transform(X_train_fs)
    X_test_wfs = fit.transform(X_test_fs)
    
    best = sfs.k_feature_names_     # to get the final set of features
    brint(best)

    return X_train_wfs, X_test_wfs

    

def get_performace_measures(model, X_train, X_test, y_train, y_test, Accuracy_list):
    model = OneVsRestClassifier(model).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy_list.append(accuracy_score(y_test, y_pred))

    yT = model.label_binarizer_.transform(y_test).toarray().T
    # Iterate through all L classifiers
    print("------------------------------")
    for i, (classifier, is_ith_class) in enumerate(zip(model.estimators_, yT)):
        print(classifier.score(X_test, is_ith_class))
        Accuracy_list[i] += classifier.score(X_test, is_ith_class)
        


# Currenlty not in use
def dimensionality_reduction(X,y):
        pca = KernelPCA(kernel='rbf')
        #pca_reduced_data = pca.fit_transform(X,y)
        X_transformed = pca.fit_transform(X)
        return X_transformed
        #lda = LinearDiscriminantAnalysis()
        #pca_lda_data = lda.fit_transform(pca_reduced_data,y)
       

X, y = read_data('../prad_tcga_genes.csv', '../prad_tcga_clinical_data.csv')

#Resampling dataset, since our data is imbalanced 
sme = SMOTEENN(random_state=42,smote=SMOTE(random_state=42, k_neighbors=1))
X, y = sme.fit_resample(X, y)
print('Resampling of dataset using SMOTEENN %s' % Counter(y), '\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state=42)

X_train_norm, X_test_norm = prepare_inputs(X_train, X_test)
X_train_fs, X_test_fs = filter_feature_selection(X_train_norm, y_train, X_test_norm, chi2)
X_train_wfs, X_test_wfs = forward_feature_selecion(X_train_fs, y_train, X_test_fs)

Accuracy_list =  [0, 0, 0, 0, 0]
get_performace_measures(MODEL, X_train_wfs, X_test_wfs, y_train, y_test, Accuracy_list)

    
print("-----------------------------------------------")
# print("Accuracy: ", round(mean(Accuracy_list), 4))
for i, a in enumerate(Accuracy_list):
    print("Accuracy of", 6+i, "vs Rest: ", round(a, 4) )
print("-----------------------------------------------")


