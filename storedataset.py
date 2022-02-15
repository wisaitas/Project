import matplotlib.pyplot as plt
import numpy as np
import random
import math
import statistics
import csv
import pandas as pd

Ptdevice = [5*(10**-3),70*(10**-3),150*(10**-3)]
SIR = []
linex = []
liney = []
radiusrangex = []
radiusrangey = []
treex = []
treey = []
tree = 0
pathlossall = []
Pr = []
dupliPr = []
Prssall = []
Prssalldbm = []
SNRall = []
dx = []
dy = []
dt = []
dtcoor = 0
coorserverx = []
coorservery = []

a = 1 # Amplitude Carrier Signal
f = 1 # Frequency Carrier Signal
Fs = 5  # Sampling rate
Ts = 1.0/Fs # Sampling interval
time = 100 # Time in graph
t = np.arange(0,time,Ts) # (start,end,between)
noise = np.random.normal(0,1,len(t)) # AWGN Noise
sig = a*np.sin(2*np.pi*f*t) # Signal Carrier Sine Wave

# Simulation Farm
sizefarm = 'normal' 
if sizefarm == "normal" or sizefarm == "Normal":
    tree = 700
    countcol = 3
    countrow = 3
    linex = [0,200,200,0,0]
    liney = [0,0,200,200,0]
    radiusrangex = [101.3,200]
    radiusrangey = [100,100]
    coorserverx = [100]
    coorservery = [100]
    ccnormal = plt.Circle((100,100),100,color='blue',fill=False)
    # fig, ax = plt.subplots()
    # ax.add_patch(ccnormal)
    #เลื่อนตามแนวแกนตั้ง
    while countrow < 200:
        #เลื่อนตามแนวแกนนอน
        while countcol < 200:
            treex.append(countcol)
            treey.append(countrow)
            eudis = ((countcol - coorserverx[0])**2 + (countrow - coorservery[0])**2)**0.5
            if eudis < 100:
                dx.append(countcol)
                dy.append(countrow)
                dt.append(eudis)
            countcol = countcol + 6.7
        countrow = countrow + 6.7
        countcol = 3

# FSPL
for i in range(len(dt)):
    cn = 0
    cn = ((math.pi)*(dt[i])*(2.4*10**9))/(3*10**8)
    if (cn != 0):
        tl = 20*math.log(cn)
        pathlossall.append(tl)

# Prss dbm
for i in Ptdevice:
    for x in range(len(dt)):
        if 8*4*math.pi*dt[x] != 0:
            if i*((1/(8*4*math.pi*dt[x]))**2) != 0:
                Pr.append(i*((1/(8*4*math.pi*dt[x]))**2)*(10**3))
                Prssalldbm.append((10 * math.log((i*((1/(8*4*math.pi*dt[x]))**2))*(10**3),10)) + 30)

# SNR dbm
for i in Prssalldbm:
    SNRall.append(i - 5)

# SIR dbm
dupliPr = Pr
below = 0
for i in range(100):
    pos = random.randint(0,len(Pr)-1)
    upeq = dupliPr.pop(pos)
    for x in dupliPr:
        below = below + x
    if(upeq / below != 0):
        SIR.append(upeq / below) 
        below = 0

# Random Binary Signal
allsig = []
amountsignal = 4
i = 0
j = 0
for s in range(amountsignal):
    ranbisig = []
    while i < len(t):
        ran = random.randint(0,1)
        if ran != 0:
            while j < 4 and i < len(t):
                ranbisig.append(1)
                i = i + 1
                j = j + 1
        else: 
            while j < 4 and i < len(t):
                ranbisig.append(0)
                i = i + 1 
                j = j + 1
        j = 0
    i = 0
    allsig.append(ranbisig)
print('------Random Binary Complete------')

# Random Binary i(t)
it = []
i = 0
j = 0
while i < len(t):
    ran = random.randint(0,1)
    if ran != 0:
        while j < 4 and i < len(t):
            it.append(0.5)
            i = i + 1
            j = j + 1
    else: 
        while j < 4 and i < len(t):
            it.append(0)
            i = i + 1 
            j = j + 1
    j = 0

# Transmited Signal 
transig = []
for item in allsig:
    caltransig = []
    for i in range(len(item)):
        caltransig.append(item[i]*sig[i])
    transig.append(caltransig)

complextransig = []
z=[]
for item in allsig:
    calcomplex = []
    for i in range(len(item)):
        h = 1/math.sqrt(2)*(random.randint(1,len(t))+1j*random.randint(1,len(t)))
        z.append(math.sqrt(h.real**2 + h.imag**2))
        calcomplex.append(item[i]*z[i] + noise[i] + SNRall[i] + it[i])
    complextransig.append(calcomplex)
print('------Transmited Signal Complete------')

# Pattern SIR dbm + complextransig
SIRpattern = []
pattern = 0
for i in range(7):
    if len(SIRpattern) < 100:
        SIRpattern.append(pattern)
        pattern = pattern + 5
    else:
        break
complextransigplussir = []
for sir in SIRpattern:
    for item in complextransig:
        calcomplexplussir = []
        for i in range(len(item)):
            calcomplexplussir.append(item[i] + sir)
        complextransigplussir.append(calcomplexplussir)
# print(complextransigplussir[0],end='\n\n\n\n\n\n')
# print(complextransigplussir[2])

# Demod ask to Transmited Signal
complextransigtoask = []
for item in complextransig:
    caltrantoask = []
    for i in range(len(item)):
        caltrantoask.append(item[i]/z[i])
    complextransigtoask.append(caltrantoask)
demodtransmited = []
for item in complextransigtoask:
    caldemod = []
    for i in range(len(item)):
        if abs(item[i]) <= abs(statistics.mean(item))/2:
            caldemod.append(0)
        else:
            caldemod.append(1)
    demodtransmited.append(caldemod)
countdemod = 0
for item in demodtransmited:
    w1 = 1
    while w1 < len(item)-1:
        if item[w1-1] == 1 and item[w1+1] == 1:
            demodtransmited[countdemod][w1] = 1
        w1 = w1+1
    countdemod = countdemod + 1
print('------demod complete------')

# Compare Transmited Sigmal before after demod
amountsignalcorrect = 0
percentcorrect = []
for item in demodtransmited:
    calcorrect = []
    countcorrect = 0
    for i in range(len(item)):
        if allsig[amountsignalcorrect][i] == demodtransmited[amountsignalcorrect][i]:
            countcorrect = countcorrect + 1
    amountsignalcorrect = amountsignalcorrect + 1
    calcorrect.append((countcorrect / len(item))*100)
    percentcorrect.append(calcorrect)
# print(percentcorrect)

# Write Dateset
time_df = []
for i in range(100):
    i = i+1
    time_df.append(['t={}'.format(i)])
df = []
dfcon = []
indcol = 0
for i in range(len(complextransigplussir)):
    if i%amountsignal == 0 and i == 0:
        df.append(pd.DataFrame(complextransigplussir[i],columns=time_df[indcol]))
    elif i%amountsignal == 0 and i != 0:
        indcol = indcol + 1
        df.append(pd.DataFrame(complextransigplussir[i],columns=time_df[indcol]))
    else:
        df.append(pd.DataFrame(complextransigplussir[i],columns=time_df[indcol]))
for i in range(len(df)):
    dfcon.append(df[i])
condf = pd.concat(dfcon,axis=1)
# condf.to_csv('signal_dataset.csv')

# Write Label
pd.Series(np.array(allsig).reshape(-1).astype(np.uint8)).to_csv('labels2.csv', index=False)

fig,myplot = plt.subplots(5, 1)
myplot[0].plot(t,allsig[0])
myplot[0].set_xlabel('Time')
myplot[0].set_ylabel('Amplitude')

myplot[1].plot(t,complextransig[0])
myplot[1].set_xlabel('Time')
myplot[1].set_ylabel('Amplitude')

myplot[2].plot(t,complextransigtoask[0])
myplot[2].set_xlabel('Time')
myplot[2].set_ylabel('Amplitude')

myplot[3].plot(t,demodtransmited[0])
myplot[3].set_xlabel('Time')
myplot[3].set_ylabel('Amplitude')

myplot[4].plot(t,complextransigplussir[0])
myplot[4].set_xlabel('Time')
myplot[4].set_ylabel('Amplitude')
plt.show()