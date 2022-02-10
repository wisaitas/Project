import matplotlib.pyplot as plt
import numpy as np
import random
import math
import statistics
import csv

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
t = np.arange(0,100,Ts) # (start,end,between)
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
ranbisig = []
i = 0
j = 0
for s in range(10):
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
transig = ranbisig * sig
complextransig = []
z = []
print('between 1 - 100 : ',len(t))
for i in range(len(t)):
    h = 1/math.sqrt(2)*(random.randint(1,len(t))+1j*random.randint(1,len(t)))
    z.append(math.sqrt(h.real**2 + h.imag**2))
    complextransig.append(transig[i]*z[i] + noise[i] + SNRall[i] + it[i])

# Pattern SIR dbm + complextransig
SIRpattern = [0,5,10,15,20,25,30]
complextransigplussir = []
for i in SIRpattern:
    for j in range(len(complextransig)):
        complextransigplussir.append(complextransig[j] + i)
print(len(complextransigplussir))


# Demod Transmited Signal + noise to ask
complextransigtoask = []
for i in range(len(t)):
    complextransigtoask.append(complextransig[i]/z[i])

# Demod ask to Transmited Signal
demodtransmited = []
for i in range(len(t)):
    if abs(complextransigtoask[i]) <= abs(statistics.mean(complextransigtoask))/2:
        demodtransmited.append(0)
    else:
        demodtransmited.append(1)
w1 = 1
while w1 < len(complextransigtoask)-1:
    if demodtransmited[w1-1] == 1 and demodtransmited[w1+1] == 1:
        demodtransmited[w1] = 1
    w1 = w1+1

# Compare Transmited Sigmal before after demod
countcorrect = 0
for i in range(len(ranbisig)):
    if ranbisig[i] == demodtransmited[i]:
        countcorrect = countcorrect + 1
percentcorrerct = (countcorrect / len(ranbisig)) * 100
print('percentcorrect = ',percentcorrerct)

# Write Dateset
f = open('signal.csv','w')
# cwr = 0
# print(complextransigplussir[len(complextransigplussir)-1])
# for row in range(7):
#     # print(row,end=' ')
#     f.write(str(row)+' ')
#     for col in range(len(t)):
#         if cwr < len(complextransigplussir):
#             # print(complextransigplussir[cwr],end=' ')
#             temp = complextransigplussir[cwr]
#             f.write(str(complextransigplussir[cwr])+' ')
#             cwr = cwr + 1
#         else:
#             break
#     # print('\n')
#     f.write('\n')
# for row in range(len(t)):
#     print(row,end=' ')
#     for col in range(len(t)):
#         p = complextransigplussir.pop(0)
#         print(p,end=' ')
#     print('\n')

fig,myplot = plt.subplots(5, 1)
myplot[0].plot(t,ranbisig)
myplot[0].set_xlabel('Time')
myplot[0].set_ylabel('Amplitude')

myplot[1].plot(t,complextransig)
myplot[1].set_xlabel('Time')
myplot[1].set_ylabel('Amplitude')

myplot[2].plot(t,complextransigtoask)
myplot[2].set_xlabel('Time')
myplot[2].set_ylabel('Amplitude')

myplot[3].plot(t,demodtransmited)
myplot[3].set_xlabel('Time')
myplot[3].set_ylabel('Amplitude')

myplot[4].plot(t,sig)
myplot[4].set_xlabel('Time')
myplot[4].set_ylabel('Amplitude')
plt.show()