import statistics as st

# [[17787  6985]
#  [ 4713 20435]]
TP,FP,FN,TN = 6766,650,528,6976
Precision = TP/(TP+FP) 
Recall = TP/(TP+FN)
F1 = 2/((1/Recall)+(1/Precision))
print('Precision = ',Precision)
print('Recall = ',Recall)
print('F1-Score = ',F1)
# meanresult = [0.7239,0.7151,0.7164,0.7274,0.7206]
# print(st.mean(meanresult))