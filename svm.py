from tabnanny import verbose
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC as SVC

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
nt = 80
for i in range(nt):
    dft.append(df.shift(i+1))
dft = pd.concat(dft, axis=1)
dft = dft.iloc[nt:]
labels = labels.iloc[nt:]
clf = SVC(kernel='rbf')
clf.fit(dft,labels)
pred = clf.predict(dft)
acc = accuracy_score(labels, pred)
print('svm accuracy : ',acc*100)
print(confusion_matrix(labels, pred))










# print('--------'*10)
# thresh = (df.iloc[nt:] > 0.5).astype(np.uint8)
# print(accuracy_score(labels, thresh))
# print(confusion_matrix(labels, thresh))