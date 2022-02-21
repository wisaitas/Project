from tabnanny import verbose
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC as SVC,SVR
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.read_csv('signal_dataset.csv',header=None)
labels = pd.read_csv('labels.csv')
labels = labels['0']
df = df.drop([0],axis=0)
df = df.drop(columns=[0],axis=1)
tmp = []
n = 4 # amount signal 
s = 28 # 7 * amount signal
for i in range(1, s, n):
    arr = df[np.arange(i, i+n)].to_numpy()
    arr = np.hstack(arr)
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
clf = SVR(kernel='linear')
clf.fit(dft,labels)
pred = clf.predict(dft)
fpr,tpr, _ = metrics.roc_curve(labels,pred)
pred_label = pred > 0.5

clf2 = SVR(kernel='rbf')
clf2.fit(dft,labels)
pred2 = clf2.predict(dft)
fpr2,tpr2, _2 = metrics.roc_curve(labels,pred2)
pred_label = pred2 > 0.5

# acc = accuracy_score(labels, pred_label)


# fpr,tpr,thresholds = metrics.roc_curve(labels,y_score)

# clf2 = SVC(kernel='rbf')
# clf2.fit(dft,labels)
# pred2 = clf2.predict(dft)
# acc2 = accuracy_score(labels, pred2)
# print('svr linear accuracy : ',acc*100 )
# print(confusion_matrix(labels, pred_label))
# print(fpr)
plt.plot(fpr,tpr,color='red',label='linear')
plt.plot(fpr2,tpr2,color='blue',label='rbf')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# print('\n')
# print('svm rbf accuracy : ',acc2*100)
# print(confusion_matrix(labels,pred2))
# tp , fn , fp , tn = np.hstack(confusion_matrix(labels,pred))











# print('--------'*10)
# thresh = (df.iloc[nt:] > 0.5).astype(np.uint8)
# print(accuracy_score(labels, thresh))
# print(confusion_matrix(labels, thresh))