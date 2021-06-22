import warnings
warnings.filterwarnings("ignore", category=Warning)
import argparse
import numbers
import pandas as pd
import numpy as np
import tensorflow as tf



def data_preprocessing(df_path, k, df_columns):
  data=pd.read_csv(df_path,error_bad_lines=False, skiprows=k,names=df_columns)
  data['Class']=data['Class'].str.strip()

  d_min=data.loc[data['Class']=='positive']
  d_maj=data.loc[data['Class']=='negative']
  print("No of minority Class:", d_min.shape[0]) #77
  print("No of majority Class:", d_maj.shape[0])
  data=np.array(data)
  X = data[:,0:-1]
  y = data[:,-1]

  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

  from sklearn.preprocessing import MinMaxScaler
  min_max_scaler = MinMaxScaler()
  X_train = min_max_scaler.fit_transform(X_train)
  X_test= min_max_scaler.transform(X_test)
  # # One-hot encode.
  from sklearn.preprocessing import LabelEncoder
  encoder = LabelEncoder()
  y_train=encoder.fit_transform(y_train)
  y_test=encoder.transform(y_test)

  d_train=pd.concat([pd.DataFrame(X_train),pd.DataFrame(y_train)], axis=1)
  d_train.columns = [*d_train.columns[:-1], 'Class']

  d_test=pd.concat([pd.DataFrame(X_test),pd.DataFrame(y_test)], axis=1)
  d_test.columns = [*d_test.columns[:-1], 'Class']
  
  return d_train, d_test