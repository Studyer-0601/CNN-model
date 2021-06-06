from rdkit import Chem as chem
from rdkit import DataStructs as datastructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
import numpy as np
import xlrd
import matplotlib.pyplot as plt
import math
from rdkit.Chem.Draw import SimilarityMaps
from pychem.pychem1 import PyChem2d
from pychem import constitution
from pychem import connectivity as co
from pychem.pychem1 import Chem
from pychem import bcut
from rdkit import Chem as chem
from rdkit import DataStructs as datastructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
import numpy as np
import xlrd
import math
from rdkit.Chem.Draw import SimilarityMaps
from pychem.pychem1 import PyChem2d
y=[]
x=[[] for _ in range(221)]
workbook=xlrd.open_workbook(r'C:\Users\Administrator\Desktop\SO2rongjiedu.xlsx')
data=workbook.sheets()[0]
YIN=data.col_values(0)
YANG=data.col_values(1)
t=data.col_values(2)
p=data.col_values(3)
y=data.col_values(4)
i=0
while i < 221:
    a=YIN[i]
    b=YANG[i]
    print(i+1)
    alldes1 = {}
    alldes2= {}
    drug1 = PyChem2d()
    drug1.ReadMolFromSmile(a)
    alldes1.update(drug1.GetAllDescriptor())
    drug2 = PyChem2d()
    drug2.ReadMolFromSmile(b)
    alldes2.update(drug2.GetAllDescriptor())
    for l in alldes1.values():
        x[i].append(l)
    for k in alldes2.values():
        x[i].append(k)
    x[i].append(t[i])
    x[i].append(p[i])
    i=i+1
import keras
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.utils import np_utils,plot_model
from sklearn.model_selection import cross_val_score,train_test_split
from keras.layers import Dense, Dropout,Flatten,Conv1D,MaxPooling1D
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from pandas.core.frame import DataFrame
x1=DataFrame(x)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from feature_selector import  FeatureSelector
fs=FeatureSelector(data=x1,labels=y)
fs.identify_missing( missing_threshold=0.1)
fs.missing_stats.head()
fs.identify_single_unique()
fs.identify_collinear(correlation_threshold=0.79)
train_no_missing = fs.remove(methods = ['missing','single_unique','collinear'],keep_one_hot=True)
train_no_missing=np.array(train_no_missing)
x1=train_no_missing.tolist()
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
x1=StandardScaler().fit_transform(x1)
y=np.array(y)
x1=np.expand_dims(x1,axis=2)
print(x1[0])
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.2)
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense,LSTM,TimeDistributed
from keras.optimizers import Adam
from keras.layers import Convolution1D,ZeroPadding1D,MaxPool1D
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPool1D
from keras.callbacks import EarlyStopping
from keras.optimizers import Nadam
model = Sequential()
model.add(Convolution1D(32, 3, activation='sigmoid', padding="same"))
model.add(Convolution1D(64, 3, activation='sigmoid'))
model.add(MaxPool1D(1))
model.add(Convolution1D(32, 3, activation='sigmoid'))
model.add(MaxPool1D(1))
model.add(Dropout(0))
model.add(Flatten())
model.add(Dense(100, activation='sigmoid', name='Dense_1'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.compile(Adam(0.006), loss='mse')
print('train=========')
early_stopping = EarlyStopping(monitor='loss', patience=50, verbose=2)
model.fit(x_train,y_train,epochs=10000,callbacks=early_stopping)
cost = model.evaluate(x_train, y_train)
print('test cost: ', cost)
print(model.summary())
from keras.models import Model
model1=Model(inputs=model.input,outputs=model.get_layer('Dense_1').output)
x_train=model1.predict(x_train)
x_test=model1.predict(x_test)
x1=model1.predict(x1)
from sklearn.ensemble import RandomForestRegressor
modelx=RandomForestRegressor(n_estimators=45)
modelx.fit(x_train,y_train)
