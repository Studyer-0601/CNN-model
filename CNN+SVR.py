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
x=[[] for _ in range(266)]
workbook=xlrd.open_workbook(r'C:\Users\Administrator\Desktop\toxic159.xlsx')#打开excel
data=workbook.sheets()[0]
yin=data.col_values(8)
yang=data.col_values(7)
t=data.col_values(2)
p=data.col_values(3)
from pychem import estate
from pychem import basak
from pychem import moran,geary
from pychem import molproperty as mp
from pychem import moe
from pychem import topology
from rdkit.ML.Descriptors import MoleculeDescriptors
#print(YIN)
from rdkit.ML.Descriptors import MoleculeDescriptors
#print(YIN)
des=[x[0] for x in Descriptors._descList]
des.pop(10)
des.pop(10)
des.pop(10)
des.pop(10)
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des)

a=calculator.GetDescriptorSummaries()
i=0
while i < 196:
     if a[i]=='N/A':
         des.pop(i)
     i=i+1
print(len(des))
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des)
i=0
while i < 266:
    print(i)
    #print(YIN[i+1])
    a=yin[i]
    b = yang[i]
    print(a)
    mol=Chem.MolFromSmiles(a)
    molb = Chem.MolFromSmiles(b)
    xx=calculator.CalcDescriptors(mol)
    xxb=calculator.CalcDescriptors(molb)
    xx=list(xx)
    xxb = list(xxb)
    x[i]=xx+xxb
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
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.2)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution1D
from tensorflow.python.keras.layers import Dense, Dropout,  Flatten,  MaxPool1D
from keras.callbacks import EarlyStopping
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
from keras.models import Model
model1=Model(inputs=model.input,outputs=model.get_layer('Dense_1').output)
x_train=model1.predict(x_train)
x_test=model1.predict(x_test)
x1=model1.predict(x1)
from sklearn.svm import SVR
modelx=SVR(kernel='rbf',C=8)
modelx.fit(x_train,y_train)
