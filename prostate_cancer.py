import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn import svm                                  
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


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
from imblearn.under_sampling import EditedNearestNeighbours 


# MODEL = svm.SVC(kernel='linear')
MODEL = GaussianNB()
# MODEL = RandomForestClassifier()
# MODEL = KNeighborsClassifier(n_neighbors=3)
# MODEL = svm.SVC(kernel='rbf')
# MODEL = LinearDiscriminantAnalysis()

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

def visualize_class_distribution(y, title):
    # Visualize the percentage of classes
    counter = Counter(y)
    label = counter.keys()
    size = counter.values()
    wedges, texts, autotexts = plt.pie(size, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.legend(wedges, label, title=title, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.axis('equal')
    plt.show()

# Dimentionaliy reduction - Not in use since it was causing low accuracy
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
    plt.show()
    
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
    
    mask = fit.get_support() #list of booleans
    best_feature_list = [] # The list of your K best features
    
    for bool, feature in zip(mask, X_train.columns):
        if bool:
            best_feature_list.append(feature)
            
    # print(pd.DataFrame(X_train_fs, columns=new_features))
    
    # DONOT REMOVE 
    # dfscores = pd.DataFrame(fit.scores_)
    # dfcolumns = pd.DataFrame(X_train.columns)
    # featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    # featureScores.columns = ['Features','Score']
    # best = featureScores.nlargest(300,'Score')
    # print(best['Features'].values.sort() == new_features.sort())
    # for f in best['Features']:
    #     featureVoting[f] = featureVoting.get(f, 0) + 1 
    
    # best.plot(x='Features', y="Score", kind="bar")
    return X_train_fs, X_test_fs, best_feature_list

def forward_feature_selecion(X_train_fs, y_train, X_test_fs, best_feature_list):
    sfs = SFS(MODEL, 
		k_features=20,
		forward=True,                     # when set false this becomes backward feature selection
		floating = False,             
        verbose=2,
		scoring = 'accuracy',
		cv = 5,
        n_jobs= -1)
    fit = sfs.fit(X_train_fs, y_train, custom_feature_names=best_feature_list)
    print(fit)
    X_train_wfs = fit.transform(X_train_fs)
    X_test_wfs = fit.transform(X_test_fs)
    
    best = sfs.k_feature_names_     # to get the final set of features
    print(best)

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
        
# Not in use since it was leading to low accuracy
def outliers_removal(X, y):
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(X)
    mask = yhat != -1
    print(mask)
    return X.iloc[mask, :], y.iloc[mask]



#step 1 read the data
X, y = read_data('../prad_tcga_genes.csv', '../prad_tcga_clinical_data.csv')

# Visualize the class distribution before oversampling
visualize_class_distribution(y, "Before Resampling")

#Resampling the training dataset, since our data is imbalanced (https://datascience.stackexchange.com/a/15633)
sme = SMOTEENN(random_state=42,smote=SMOTE(random_state=42, k_neighbors=2))
X, y = sme.fit_resample(X, y)
print('Resampling of dataset using SMOTEENN %s' % Counter(y), '\n')

print(X.shape, y.shape)

# Visualize the class distribution after Oversampling
visualize_class_distribution(y, "After Resampling")

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state=42)
X_train_norm, X_test_norm = prepare_inputs(X_train, X_test)
X_train_fs, X_test_fs, best_feature_list = filter_feature_selection(X_train_norm, y_train, X_test_norm, mutual_info_classif)
X_train_wfs, X_test_wfs = forward_feature_selecion(X_train_fs, y_train, X_test_fs, best_feature_list)

Accuracy_list =  [0, 0, 0, 0, 0]
get_performace_measures(MODEL, X_train_wfs, X_test_wfs, y_train, y_test, Accuracy_list)
    
print("-----------------------------------------------")
# print("Accuracy: ", round(mean(Accuracy_list), 4))
for i, a in enumerate(Accuracy_list):
    print("Accuracy of", 6+i, "vs Rest: ", round(a, 4) )
print("-----------------------------------------------")


