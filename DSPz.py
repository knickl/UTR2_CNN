#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np
from datetime import datetime
from datetime import date
from datetime import timedelta
import os
import cv2
import matplotlib.pyplot as plt


# In[2]:


fname = '/media/WDSSD/MachineLearning/UTR-2/L160612_045216.jds'


# In[3]:


fname = '/media/WDSSD/MachineLearning/UTR-2/L160612_045216.jds'


# In[4]:


fname


# In[7]:


with open(fname, "rb") as file:
    arr = file.read()


# In[8]:


data_block = np.frombuffer(arr, dtype='int32', count=-1, offset=1024)


# In[9]:


data_block


# In[18]:


item = np.frombuffer(arr, dtype = 'S1', count = 32, offset = 32)


# In[21]:


item


# In[19]:


name = "".join([b.decode("ascii") for b in item if b != b'\n'])


# In[59]:


#read header
hdr_list = [32]*5 + [96, 256, 64] + [4]*32
for idx in range(len(hdr_list)):
    byte_start = sum(hdr_list[0:idx])

    if idx == 0:
        item = np.frombuffer(arr, dtype='S1', count=hdr_list[idx], offset=byte_start)
        name = "".join([b.decode("ascii") for b in item if b != b'\n'])
    elif idx == 1:
        item = np.frombuffer(arr, dtype='S1', count=hdr_list[idx], offset=byte_start)
        time = "".join([b.decode("ascii") for b in item if b != b'\n'])
        dtfmt = '%a %b %d %H:%M:%S %Y'
        dttime = datetime.strptime(time, dtfmt)     
    elif idx == 2:
        item = np.frombuffer(arr, dtype='S1', count=hdr_list[idx], offset=byte_start)
        gmtt = "".join([b.decode("ascii") for b in item if b != b'\n'])
    elif idx == 3:
        item = np.frombuffer(arr, dtype='S1', count=hdr_list[idx], offset=byte_start)
        sysn = "".join([b.decode("ascii") for b in item if b != b'\n'])
    elif idx == 4:
        syst = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 5:
        item = np.frombuffer(arr, dtype='S1', count=hdr_list[idx], offset=byte_start)
        place = "".join([b.decode("ascii") for b in item if b != b'\n'])
    elif idx == 6:
        item = np.frombuffer(arr, dtype='S1', count=hdr_list[idx], offset=byte_start)
        desc = "".join([b.decode("ascii") for b in item if b != b'\n'])
    elif idx == 7:        
        PP = np.frombuffer(arr, dtype='int32', count=16, offset=byte_start)
        packT = PP[3]
        packF = PP[4]
    elif idx == 8:
        FFT_Size = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 9:
        MinDSPSize = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 10:
        MinDMASize = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 11: 
        DMASizeCnt = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 12: 
        DMASize = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 13:
        CLCfrq = np.frombuffer(arr, dtype='float32', count=1, offset=byte_start)
    elif idx == 14:
        Synch = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 15:
        SSht = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 16:
        Mode = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 17:
        Wch = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 18:
        Smd = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 19:
        Offt = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
        Offt = np.asscalar(Offt)
    elif idx == 20:
        Lb = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 21:
        Hb = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 22:
        Wb = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 23:
        NAvr = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    #elif idx == 24:
        #packT = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    #elif idx == 25:
        #packF = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 26:
        DCRem = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 27:
        ExtSyn = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 28:
        Ch1 = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 29:
        Ch2 = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 30:
        ExtWin = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 31:
        Clip = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 32:
        HPF0 = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 33:
        HPF1 = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 34:
        LPF0 = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 35:
        LPF1 = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 36:
        ATT0 = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)
    elif idx == 37:
        ATT1 = np.frombuffer(arr, dtype='int32', count=1, offset=byte_start)

#read Data Block
data_block = np.frombuffer(arr, dtype='int32', count=-1, offset=1024)

sps = CLCfrq # ADC sampling frequency
df = sps*packF/16384    # actual frequency resolution
widths = np.asarray([8192,4096,4096,Wb]) # original widths of spectrum in frequency channels
Nmin = np.asarray([0,0,4096, Lb])
Fmin=Nmin[Offt]*df/packF

Nf= widths[Offt]/packF
Nf = Nf.astype(int)

FW=Nf*df # the widths of frequency band of observations
STS = Nf*2*4 #Size of one Time Sample
dt = 8192/sps*NAvr*packT # actual time resolution

File_Size = os.path.getsize(fname)

TNS=(File_Size-1024)/STS # Total Number of Samples
TNS = np.floor(TNS)
TNS = TNS.astype(int)
TT=TNS*dt #Total Time covered by file
data_block = data_block[0:TNS[0]*2*Nf[0]] # 

data = np.transpose(data_block.reshape(TNS[0],2*Nf[0]))
PAB = data[0::2,:]/100
POL = data[1::2,:]/10000

#fig, (ax0) = plt.subplots(nrows=1)
#pc0 = ax0.pcolormesh(PAB,cmap='gist_ncar_r')
#plt.show()

Fmin = Fmin/1000000
Fmax = Fmin + FW/1000000

sTime = dttime.strftime("%H:%M:%S")
eTime = dttime + timedelta(seconds=TT[0])
eTime = eTime.strftime("%H:%M:%S")


# In[70]:


img = cv2.resize(PAB, (2*480,2*640))


# In[74]:


packT


# In[72]:


title =name + ' ' + time 
fig, (ax0) = plt.subplots(nrows=1)
pc0 = ax0.pcolormesh(img,cmap='gist_ncar')
ax0.set_title(title)
ax0.set_xticklabels([])
ax0.set_xticks(np.linspace(0,2*480,5,endpoint = 'True'))
ax0.set_xticklabels([sTime,(dttime + timedelta(hours=2)).strftime("%H:%M:%S"),(dttime + timedelta(hours=4)).strftime("%H:%M:%S"),(dttime + timedelta(hours=6)).strftime("%H:%M:%S"),eTime])
ax0.set_yticklabels([])
ax0.set_yticks(np.linspace(0,2*640,5,endpoint = 'True'))
ax0.set_yticklabels(np.round(np.linspace(Fmin, Fmax, num=5, endpoint=True, retstep=False, dtype=None)))
ax0.set_ylabel('Frequency [MHz]')
ax0.set_xlabel('Time [Hour of Day]')
fig.colorbar(pc0, ax=ax0,label='[db] above BG') 
plt.show()


# In[75]:


fname = 'ölaksjdf/asdf'


# In[86]:


fname = '/media/WDSSD/MachineLearning/UTR-2/test'
f  = open(fname, 'w')


# In[87]:


f.write('asdf\n')


# In[88]:


f.write('qwertzuiop\n')


# In[89]:


f.close()


# In[ ]:





# In[ ]:




