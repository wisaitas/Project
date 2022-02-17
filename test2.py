from tabnanny import verbose
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC as SVC
from sklearn.model_selection import train_test_split

df = pd.read_csv('signal_dataset.csv',header=None)
labels = pd.read_csv('labels.csv')
labels = labels['0']
df = df.drop([0],axis=0)
df = df.drop(columns=[0],axis=1)
tmp = []
n = 100 # amount signal 
s = 700 # amount SIR * amount signal
for i in range(1, s, n):
    arr = df[np.arange(i, i+n)].to_numpy()
    arr = np.hstack(arr)
    # print(arr.shape)
    tmp.append(pd.Series(arr))

df = pd.DataFrame(tmp).T
df = df.astype(float)
df = df[0]
df = (df - df.min()) / (df.max() - df.min())
dft = [df.copy()]
nt = 90
for i in range(nt):
    dft.append(df.shift(i+1))
# print(df, df.shift(1))
dft = pd.concat(dft, axis=1)
dft = dft.iloc[nt:]
labels = labels.iloc[nt:]
# X_train, X_test, y_train, y_test = train_test_split(dft, labels, test_size=0.33, random_state=42)
# clf = SVC(kernel='rbf')
# # print(df.shape, labels.shape)
# clf.fit(dft,labels)
# pred = clf.predict(dft)
# acc = accuracy_score(labels, pred)
# print('svm accuracy : ',acc*100)
# print(confusion_matrix(labels, pred))