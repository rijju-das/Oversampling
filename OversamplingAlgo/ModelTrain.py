import numpy
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
import sklearn.metrics as metrics
#from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.constraints import maxnorm
from sklearn.model_selection import cross_validate
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


class ModelTrain(object):
  # def __init__(self,k):
  #   self.k=k
  def train(self,X_train, y_train):  
      models_p = []
      models_p.append(('RFC', RandomForestClassifier(n_estimators=100,random_state=0)))
      models_p.append(('LDA', LinearDiscriminantAnalysis()))
      models_p.append(('KNN', KNeighborsClassifier()))
      models_p.append(('CART', DecisionTreeClassifier()))
      models_p.append(('NB', GaussianNB()))
      models_p.append(('SVM', SVC(kernel='linear')))
      # models_p.append(('L_SVM', LinearSVC()))
      models_p.append(('ETC', ExtraTreesClassifier()))
      # models_p.append(('RFC', RandomForestClassifier()))
      
      results = {}
      for name, model in models_p:
          temp=[]
          model.fit(X_train, y_train)
          y_pred = model.predict(X_test) 
          temp.append([accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, average='weighted'),recall_score(y_test, y_pred, average='weighted'),f1_score(y_test, y_pred, average='weighted')])
          # Evaluate the model
          # kfold = KFold(n_splits = 5, shuffle=True)
          #kfold = cross_validation.KFold(n=num_instances, n_folds=10, random_state=seed)
          # cv_results = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
          # results.append(cv_results)
          # names.append(name)
          # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
          # print(msg)
          results[model]=temp
      # new=pandas.DataFrame(outputs)
      # new.to_csv('pima.csv')