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
n = 3 # amount signal 
s = 7*n # 7 * amount signal
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

# clf = SVR(kernel='linear')
# clf.fit(dft,labels)
# pred = clf.predict(dft)
# fpr,tpr, _ = metrics.roc_curve(labels,pred)
# pred_label = pred > 0.5
# print('linear success')

# clfmmse = SVC(kernel='linear')
# clfmmse.fit(dft,labels)
# predmmse = clfmmse.predict(dft)
# fprmmse,tprmmse, _ = metrics.roc_curve(labels,predmmse)
# pred_labelmmse = predmmse > 0.5
# print('mmse success')

# clfrbf = SVR(kernel='rbf')
# clfrbf.fit(dft,labels)
# predrbf = clfrbf.predict(dft)
# fprrbf,tprrbf, _2 = metrics.roc_curve(labels,predrbf)
# pred_labelrbf = predrbf > 0.5
# print('rbf success')

# clfpoly2 = SVR(kernel='poly',degree = 30)
# clfpoly2.fit(dft,labels)
# predpoly2 = clfpoly2.predict(dft)
# fprpoly2,tprpoly2, _2 = metrics.roc_curve(labels,predpoly2)
# pred_labelpoly2 = predpoly2> 0.5
# print('poly degree 2 success')

clfpoly = SVR(kernel='poly',degree = 90)
clfpoly.fit(dft,labels)
predpoly = clfpoly.predict(dft)
fprpoly,tprpoly, _2 = metrics.roc_curve(labels,predpoly)
pred_labelpoly = predpoly > 0.5
print('poly degree 3 success')


# clfpoly4 = SVR(kernel='poly',degree = 40)
# clfpoly4.fit(dft,labels)
# predpoly4 = clfpoly4.predict(dft)
# fprpoly4,tprpoly4, _2 = metrics.roc_curve(labels,predpoly4)
# pred_labelpoly4 = predpoly4 > 0.5
# print('poly degree 4 success')

# acc = accuracy_score('acc linear : ',labels, pred)
# accmmse = accuracy_score('acc mmse : ',labels,pred_labelmmse)
# accrbf = accuracy_score('acc rbf : ',labels,pred_labelrbf)
# accpoly = accuracy_score('acc poly : ',labels,pred_labelpoly)
# accpoly2 = accuracy_score('acc poly : ',labels,pred_labelpoly2)
# accpoly4 = accuracy_score('acc poly : ',labels,pred_labelpoly4)


# # fpr,tpr,thresholds = metrics.roc_curve(labels,y_score)

# # clf2 = SVC(kernel='rbf')
# # clf2.fit(dft,labels)
# # pred2 = clf2.predict(dft)
# # acc2 = accuracy_score(labels, pred2)
# print('linear accuracy : ',acc*100 )
# print('RBF accuracy : ',accrbf*100 )
# print('Poly accuracy : ',accpoly*100)
# print('Poly degree 2 accuracy : ',accpoly2*100)
# print('Poly degree 4 accuracy : ',accpoly4*100)
# print(confusion_matrix(labels, pred_label))
# plt.plot(fpr,tpr,color='red',label='linear')
# plt.plot(fprmmse,tprmmse,color='blue',label='mmse ')
# plt.plot(fprrbf,tprrbf,color='c',label='rbf')
# plt.plot(fprpoly2,tprpoly2,color='m',label='poly degree 11')
plt.plot(fprpoly,tprpoly,color='green',label='poly degree 90')
# plt.plot(fprpoly4,tprpoly4,color='y',label='poly degree 13')
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