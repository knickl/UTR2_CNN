#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt


#fname = '/media/WDSSD/MachineLearning/UTR-2/UTR-2 2002_DATA/Data/test_dataframe (Kopie).npy'

fname = '/media/stefan/169B881B5E8ED996/MachineLearning/UTR-2/UTR-2 2002_DATA/Data/test_dataframe.npy'

X = np.load(fname,allow_pickle=True)


DSP_array = X[6]['DSP'][:,0:8000]
fig, ax = plt.subplots(1)
plt.rcParams["figure.figsize"] = (30,10)
ax.imshow(np.flipud(DSP_array),cmap='jet')  
plt.show()


DSP_array = X[5]['DSP'][:,0:9000]
fig, ax = plt.subplots(1)
ax.imshow(np.flipud(DSP_array),cmap='jet')  
plt.show()


# In[4]:


from datetime import datetime
from datetime import timedelta

def tvec(start_time,start_IDX,end_IDX):
    dtfmt='%d.%m.%Y %H.%M.%S'
    import pandas as pd

    a = datetime.utcfromtimestamp(int(start_time)) + timedelta(seconds=start_IDX/10)
    b = datetime.utcfromtimestamp(int(start_time)) + timedelta(seconds=end_IDX/10)
    
    start = pd.Timestamp(a)
    end = pd.Timestamp(b)
    t = np.linspace(start.value, end.value, 11)
    t = pd.to_datetime(t)
    TimeVector = []
    for i in t:
        i.to_pydatetime()
        i = i.strftime('%H:%M:%S')
        TimeVector.append(i)
    return TimeVector


# In[5]:


start_IDX = 4600
end_IDX = 8200


# In[6]:


#fname = '/media/WDSSD/MachineLearning/UTR-2/UTR-2 2002_DATA/Data/test_dataframe.npy'

fname = '/media/stefan/169B881B5E8ED996/MachineLearning/UTR-2/UTR-2 2002_DATA/Data/test_dataframe.npy'

X = np.load(fname,allow_pickle=True)
DSP = X[1]['DSP'][:,start_IDX:end_IDX]


# In[5]:


r = ((29.8211-17.3333)/1023) # the ratio of how much a pixel is worth in MHz
z = [i - 17.3333 for i in [28,26,24,22,20,18]]
frq_idx = [zi / r for zi in z]
frq_idx = [xi + 1023 - 873.8155719982702 - 54 for xi in frq_idx]
frq_idx = np.round(frq_idx,4)


# In[ ]:





# In[6]:


a = np.arange(149.8-(3*abs((np.gradient(frq_idx)/4)[0])),969+(1*abs((np.gradient(frq_idx)/4)[0])),abs((np.gradient(frq_idx)/4)[0]))
frq_idx_minor = np.round(a,4)
print(frq_idx_minor)


# In[7]:


plt.rcParams["figure.figsize"] = (30,10)
fig, ax = plt.subplots(1)
ax.imshow(np.flipud(DSP),cmap='jet')  
plt.rcParams["figure.figsize"] = (30,10)

ax.set_xticks(np.arange(0,DSP.shape[1],350))
ax.set_xticks(np.arange(0,DSP.shape[1],50), minor=True)

r = ((29.8211-17.3333)/1023) # the ratio of how much a pixel is worth in MHz
z = [i - 17.3333 for i in [28,26,24,22,20,18]]
frq_idx = [zi / r for zi in z]
frq_idx = [xi + 1023 - 873.8155719982702 - 54 for xi in frq_idx]
frq_idx = np.round(frq_idx,4)

ax.set_yticks(frq_idx)
ax.set_yticks(frq_idx_minor, minor=True)

frq = [18,20,22,24,26,28]
ax.set_yticklabels(frq)
ax.set_ylabel('Frequency [MHz]')
ax.set_xticklabels([])
ax.set_xticklabels(tvec(X[1]['start_time'],start_IDX, end_IDX))
ax.set_xlabel('Time [HH:MM:SS]')

fname = '/home/stefan/Dokumente/EPSC_2019/Images/Final/Burst.png'

plt.savefig(fname, dpi=400, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None) 

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:


import numpy as np
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Input,Flatten,Activation
from keras.layers import GlobalMaxPooling2D
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm 


MODEL_NAME = 'UTR_CNN-V3-full_frame.model'
if os.path.exists('/home/stefan/Dokumente/MachineLearning/UTR2_Python/' + MODEL_NAME):
    model = load_model(MODEL_NAME)
    print('Loaded: ' + MODEL_NAME)

fname = '/media/WDSSD/MachineLearning/UTR-2/UTR-2 2002_DATA/Data/test_dataframe.npy'
X = np.load(fname,allow_pickle=True)
#DSP = X[1]['DSP'][:,1000:8000]
DSP = X[1]['DSP'][:,4600:8200]


#fname = '/media/WDSSD/MachineLearning/UTR-2/UTR-2 2002_DATA/Data/test_dataframe (Kopie).npy'
#X = np.load(fname,allow_pickle=True)
#DSP = X[6]['DSP'][:,0:8000]


#fname = '/media/WDSSD/MachineLearning/UTR-2/UTR-2 2002_DATA/Data/test_dataframe (Kopie).npy'
#X = np.load(fname,allow_pickle=True)
#DSP = X[5]['DSP'][:,0:9000]


amax = np.amax(DSP)
amin = np.amin(DSP)

stop_f = DSP.shape[0] # maximum frequency index 
stop_t = DSP.shape[1] # maximum time index

ratio = [[500,50],[500,100],[500,200],[500,300]]
classnum = 2
rationum = len(ratio)

result_array = np.zeros((stop_f+1, stop_t+1,rationum,classnum))

result_array_full = np.zeros((stop_f+1, stop_t+1,classnum))

for r,ri in tqdm(enumerate(ratio)):   
    delta_f = round(ri[0]/2) # half window height
    delta_t = round(ri[1]/2) # half window width

    step_f = round(ri[0]/2)
    f = np.arange(0, stop_f, step_f, dtype=None)
    step_t = round(ri[1])
    t = np.arange(0, stop_t, step_t, dtype=None)
    
    for i in f:
        for j in t:
            if i-delta_f >= 0 and i+delta_f <= stop_f and j-delta_t >= 0 and j+delta_t <=stop_t:
                box = DSP[i-delta_f:i+delta_f,j-delta_t:j+delta_t]            
            elif i-delta_f < 0 and j-delta_t >= 0 and j+delta_t <=stop_t:
                box = DSP[0:i+delta_f,j-delta_t:j+delta_t]
            elif i-delta_f < 0 and j-delta_t < 0:
                box = DSP[0:i+delta_f,0:j+delta_t]
            elif i-delta_f >= 0 and i+delta_f <= stop_f and j-delta_t < 0:  
                box = DSP[i-delta_f:i+delta_f,0:j+delta_t]
            elif i+delta_f > stop_f and j-delta_t >= 0 and j+delta_t <=stop_t:
                box = DSP[i-delta_f:stop_f,j-delta_t:j+delta_t]   
            elif i+delta_f > stop_f and j+delta_t >stop_t:
                box = DSP[i-delta_f:stop_f,j-delta_t:stop_t]   
            elif i-delta_f >= 0 and i+delta_f <= stop_f and j+delta_t >stop_t:  
                box = DSP[i-delta_f:i+delta_f,j-delta_t:stop_t]
            elif i-delta_f >= 0 and i+delta_f > stop_f and j-delta_t < 0:
                box = DSP[i-delta_f:stop_f,0:j+delta_t]
            elif i-delta_f < 0 and j+delta_t > stop_t:
                box = DSP[0:i+delta_f,j-delta_t:stop_t]

            if box.size:
                cv2box = cv2.resize(box.astype('float64'), (256,256), interpolation = cv2.INTER_CUBIC)
                cv2box = (cv2box - amin) / (amax - amin)
                cv2box = cv2box.reshape((-1, 256, 256, 1))

                
                #print(r,ri,i,j)
                pred = model.predict(cv2box)
                #print(pred)
                #result_array[i-delta_t:i+delta_t,j-delta_f:j+delta_f,0] = result_array[i-delta_t:i+delta_t,j-delta_f:j+delta_f,0] + pred[0][0]
                #result_array[i-delta_t:i+delta_t,j-delta_f:j+delta_f,1] = result_array[i-delta_t:i+delta_t,j-delta_f:j+delta_f,1] + pred[0][1]
                #result_array[i-delta_t:i+delta_t,j-delta_f:j+delta_f,2] = result_array[i-delta_t:i+delta_t,j-delta_f:j+delta_f,2] + pred[0][2]
                result_array[i,j,r,0] = pred[0][0]
                result_array[i,j,r,1] = pred[0][1]
                
                #result_array_full[i-delta_t:i+delta_t,j-delta_f:j+delta_f,0] = result_array_full[i-delta_t:i+delta_t,j-delta_f:j+delta_f,0] + pred[0][0]
                #result_array_full[i-delta_t:i+delta_t,j-delta_f:j+delta_f,1] = result_array_full[i-delta_t:i+delta_t,j-delta_f:j+delta_f,1] + pred[0][1]
                
                
print(result_array[:,:,:,:].shape)


# In[9]:


import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2

fig, ax = plt.subplots(1)
ax.imshow(np.flipud(DSP),cmap='jet')

plt.rcParams["figure.figsize"] = (30,10)

#ax.pcolormesh(DSP,cmap='jet')

#plt.rcParams['figure.figsize'] = [100, 50]
layer = 1 # 0 BG, 1 Burst, 2 Bright Burst

r = 0
color = 'r'
a = np.where(result_array[251:,:,r,layer] > 0.65)
for idx in range(len(a[0])):
    rect = patches.Rectangle((a[1][idx]-ratio[r][1]/2,1020-a[0][idx]-ratio[r][0]/2),ratio[r][1],ratio[r][0],linewidth=1,edgecolor=color,facecolor='none')
    ax.add_patch(rect)
        
r = 1
color = 'orange'
a = np.where(result_array[251:,:,r,layer] > 0.74)
for idx in range(len(a[0])):
    idx
    rect = patches.Rectangle((a[1][idx]-ratio[r][1]/2,1020-a[0][idx]-ratio[r][0]/2),ratio[r][1],ratio[r][0],linewidth=1,edgecolor=color,facecolor='none')
    ax.add_patch(rect)

r = 2
color = 'g'
a = np.where(result_array[251:,:,r,layer] > 0.76)
for idx in range(len(a[0])):
    idx
    rect = patches.Rectangle((a[1][idx]-ratio[r][1]/2,1020-a[0][idx]-ratio[r][0]/2),ratio[r][1],ratio[r][0],linewidth=3,edgecolor=color,facecolor='none')
    ax.add_patch(rect)
    
r = 3
color = 'yellow'
a = np.where(result_array[251:,:,r,layer] > 0.78)
for idx in range(len(a[0])):
    idx
    rect = patches.Rectangle((a[1][idx]-ratio[r][1]/2,1020-a[0][idx]-ratio[r][0]/2),ratio[r][1],ratio[r][0],linewidth=3,edgecolor=color,facecolor='none')
    ax.add_patch(rect)    

    
ax.set_xticks(np.arange(0,DSP.shape[1],350))
ax.set_xticks(np.arange(0,DSP.shape[1],50), minor=True)

r = ((29.8211-17.3333)/1023) # the ratio of how much a pixel is worth in MHz
z = [i - 17.3333 for i in [28,26,24,22,20,18]]
frq_idx = [zi / r for zi in z]
frq_idx = [xi + 1023 - 873.8155719982702 - 54 for xi in frq_idx]
frq_idx = np.round(frq_idx,4)

ax.set_yticks(frq_idx)
ax.set_yticks(frq_idx_minor, minor=True)

frq = [18,20,22,24,26,28]
ax.set_yticklabels(frq)
ax.set_ylabel('Frequency [MHz]')
ax.set_xticklabels([])
ax.set_xticklabels(tvec(X[1]['start_time'],start_IDX, end_IDX))
ax.set_xlabel('Time [HH:MM:SS]')

fname = '/home/stefan/Dokumente/EPSC_2019/Images/Final/ROI_Burst.png'

plt.savefig(fname, dpi=400, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)    
plt.show()


# In[ ]:





# In[10]:


ratio


# In[11]:


def nonzero_intervals(vec):
    '''
    Find islands of non-zeros in the vector vec
    '''
    if len(vec)==0:
        return []
    elif not isinstance(vec, np.ndarray):
        vec = np.array(vec)

    edges, = np.nonzero(np.diff((vec==0)*1))
    edge_vec = [edges+1]
    if vec[0] != 0:
        edge_vec.insert(0, [0])
    if vec[-1] != 0:
        edge_vec.append([len(vec)])
    edges = np.concatenate(edge_vec)
    return zip(edges[::2], edges[1::2]-1)


# In[12]:


r = 0
layer = 1

step_f = round(ratio[r][0]/2)-1
step_t = round(ratio[r][1]/2)


master_shape = np.zeros((result_array.shape[0],result_array.shape[1]))


a = np.where(result_array[251:,:,r,layer] > 0.65)
arr = np.zeros((result_array.shape[0],result_array.shape[1]))
for i in range(len(a[0])):
    
    p = result_array[251:,:,r,layer][a[0][i],a[1][i]]
    
    #x0 = arr[a[0][i]-step_f:a[0][i]+step_f,a[1][i]-step_t:a[1][i]+step_t]
    #parr = [[(e + p)/2 if sum(l) else p for e in l] for l in x0]  
    #arr[a[0][i]-step_f:a[0][i]+step_f,a[1][i]-step_t:a[1][i]+step_t] = parr
    
    arr[a[0][i]-step_f:a[0][i]+step_f,a[1][i]-step_t:a[1][i]+step_t] = p
    
    master_shape[a[0][i]-step_f:a[0][i]+step_f,a[1][i]-step_t:a[1][i]+step_t] = 1
    
master_shape_boxes = master_shape[0,:]
idx_range = [list(t) for t in nonzero_intervals(master_shape_boxes)]
    
fig, ax = plt.subplots(1)
ax.imshow(np.flipud(arr),cmap='jet')
plt.show   

r = 1
layer = 1

step_f = round(ratio[r][0]/2)-1
step_t = round(ratio[r][1]/2)

a = np.where(result_array[251:,:,r,layer] > 0.74)
arr = np.zeros((result_array.shape[0],result_array.shape[1]))
for i in range(len(a[0])):
    
    p = result_array[251:,:,r,layer][a[0][i],a[1][i]] 
    
    #x0 = arr[a[0][i]-step_f:a[0][i]+step_f,a[1][i]-step_t:a[1][i]+step_t]
    #parr = [[(e + p)/2 if sum(l) else p for e in l] for l in x0]  
    #arr[a[0][i]-step_f:a[0][i]+step_f,a[1][i]-step_t:a[1][i]+step_t] = parr
    
    arr[a[0][i]-step_f:a[0][i]+step_f,a[1][i]-step_t:a[1][i]+step_t] = p
    master_shape[a[0][i]-step_f:a[0][i]+step_f,a[1][i]-step_t:a[1][i]+step_t] = 1
    
fig, ax = plt.subplots(1)
ax.imshow(np.flipud(arr),cmap='jet')

plt.show

r = 2
layer = 1

step_f = round(ratio[r][0]/2)-1
step_t = round(ratio[r][1]/2)

a = np.where(result_array[251:,:,r,layer] > 0.76)
arr = np.zeros((result_array.shape[0],result_array.shape[1]))
for i in range(len(a[0])):
    
    p = result_array[251:,:,r,layer][a[0][i],a[1][i]]
    
    #x0 = arr[a[0][i]-step_f:a[0][i]+step_f,a[1][i]-step_t:a[1][i]+step_t]
    #parr = [[(e + p)/2 if sum(l) else p for e in l] for l in x0]  
    #arr[a[0][i]-step_f:a[0][i]+step_f,a[1][i]-step_t:a[1][i]+step_t] = parr
    
    arr[a[0][i]-step_f:a[0][i]+step_f,a[1][i]-step_t:a[1][i]+step_t] = p
    master_shape[a[0][i]-step_f:a[0][i]+step_f,a[1][i]-step_t:a[1][i]+step_t] = 1
fig, ax = plt.subplots(1)
ax.imshow(np.flipud(arr),cmap='jet')
plt.show


r = 3
layer = 1

step_f = round(ratio[r][0]/2)-1
step_t = round(ratio[r][1]/2)

a = np.where(result_array[251:,:,r,layer] > 0.78)
arr = np.zeros((result_array.shape[0],result_array.shape[1]))
for i in range(len(a[0])):
    p = result_array[251:,:,r,layer][a[0][i],a[1][i]]    
    
    #x0 = arr[a[0][i]-step_f:a[0][i]+step_f,a[1][i]-step_t:a[1][i]+step_t]
    #parr = [[(e + p)/2 if sum(l) else p for e in l] for l in x0]  
    #arr[a[0][i]-step_f:a[0][i]+step_f,a[1][i]-step_t:a[1][i]+step_t] = parr
    
    arr[a[0][i]-step_f:a[0][i]+step_f,a[1][i]-step_t:a[1][i]+step_t] = p
    master_shape[a[0][i]-step_f:a[0][i]+step_f,a[1][i]-step_t:a[1][i]+step_t] = 1
fig, ax = plt.subplots(1)
ax.imshow(np.flipud(arr),cmap='jet')
plt.show


fig, ax = plt.subplots(1)
plt.rcParams["figure.figsize"] = (30,10)

ax.imshow(np.flipud(master_shape),cmap='binary')

idx_range_2 = [list(t) for t in nonzero_intervals(master_shape_boxes)]

idx_range_fin = []
for i in idx_range_2:
    c = []
    for s in idx_range:
        if i[0]<=s[0] and i[1]>=s[1]:
            c.append(1)
    t = []
    if len(c) == 0 or len(c) == 1:
        idx_range_fin.append(i)
    elif len(c) == 2:
        for s in idx_range:
            if i[0]<=s[0] and i[1]>=s[1]:
                t.append(s)
        for z in range(len(t)):
            if z == 0:
                u = [i[0],t[0][1]]
                idx_range_fin.append(u)
            elif z == 1:
                u = [t[0][0], i[1]]
                idx_range_fin.append(u)
    elif len(c) > 2:
        for s in idx_range:
            if i[0]<=s[0] and i[1]>=s[1]:
                t.append(s)
        for z in range(len(t)):
            if z == 0:
                u = [i[0],t[0][1]]
                idx_range_fin.append(u) 
            elif z == len(t)-1:
                u = [t[2][0], i[1]]
                idx_range_fin.append(u)
            else:
                u = [t[z-1][1]+1,t[z+1][0]-1]
                idx_range_fin.append(u)

#idx_range_fin_padded = []
#for i in idx_range_fin:
#    idx_range_fin_padded.append([i[0]-25,i[1]+25])                
               
for i in idx_range_fin:  
    rect = patches.Rectangle((i[0],0),i[1]-i[0],1023,linewidth=3,edgecolor='g',facecolor='none')
    ax.add_patch(rect)  
    
    
    
ax.set_xticks(np.arange(0,DSP.shape[1],350))
ax.set_xticks(np.arange(0,DSP.shape[1],50), minor=True)

r = ((29.8211-17.3333)/1023) # the ratio of how much a pixel is worth in MHz
z = [i - 17.3333 for i in [28,26,24,22,20,18]]
frq_idx = [zi / r for zi in z]
frq_idx = [xi + 1023 - 873.8155719982702 - 54 for xi in frq_idx]
frq_idx = np.round(frq_idx,4)

ax.set_yticks(frq_idx)
ax.set_yticks(frq_idx_minor, minor=True)

frq = [18,20,22,24,26,28]
ax.set_yticklabels(frq)
ax.set_ylabel('Frequency [MHz]')
ax.set_xticklabels([])
ax.set_xticklabels(tvec(X[1]['start_time'],start_IDX, end_IDX))
ax.set_xlabel('Time [HH:MM:SS]')

fname = '/home/stefan/Dokumente/EPSC_2019/Images/Final/master_shape.png'

plt.savefig(fname, dpi=400, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)
    
plt.show()    


# In[13]:


idx_range_fin


# In[ ]:





# In[14]:


DSPxarr = np.zeros((DSP.shape[0],DSP.shape[1]))
DSP_n = (DSP - amin) / (amax - amin)
for i in idx_range_fin:
    DSPxarr[:,i[0]:i[1]] = DSP_n[:,i[0]:i[1]]


# In[15]:


fig, ax = plt.subplots(1)
ax.imshow(np.flipud(DSPxarr),cmap='jet')
for i in idx_range_fin:  
    rect = patches.Rectangle((i[0],0),i[1]-i[0],1023,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)   
plt.show()


# In[16]:


'''
import numpy as np
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Input,Flatten,Activation
from keras.layers import GlobalMaxPooling2D
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm 


MODEL_NAME = 'UTR_CNN-V3-full_frame.model'
if os.path.exists('/home/stefan/Dokumente/MachineLearning/UTR2_Python/' + MODEL_NAME):
    model = load_model(MODEL_NAME)
    print('Loaded: ' + MODEL_NAME)
'''
amax = np.amax(DSPxarr)
amin = np.amin(DSPxarr)

stop_f = DSPxarr.shape[0] # maximum frequency index 
stop_t = DSPxarr.shape[1] # maximum time index

ratio = [[500,50],[250,25]]
classnum = 2
rationum = len(ratio)

result_array = np.zeros((stop_f+1, stop_t+1,rationum,classnum))

result_array_full = np.zeros((stop_f+1, stop_t+1,rationum,classnum))

for i_rf in tqdm(idx_range_fin):

    DSP = DSPxarr[:,i_rf[0]:i_rf[1]]
    for r,ri in enumerate(ratio):   
        delta_f = round(ri[0]/2) # half window height
        delta_t = round(ri[1]/2) # half window width

        step_f = round(ri[0]/2)
        f = np.arange(0, stop_f, step_f, dtype=None)
        step_t = round(ri[1])
        t = np.arange(0, stop_t, step_t, dtype=None)

        for i in f:
            for j in t:
                if i-delta_f >= 0 and i+delta_f <= stop_f and j-delta_t >= 0 and j+delta_t <=stop_t:
                    box = DSP[i-delta_f:i+delta_f,j-delta_t:j+delta_t]            
                elif i-delta_f < 0 and j-delta_t >= 0 and j+delta_t <=stop_t:
                    box = DSP[0:i+delta_f,j-delta_t:j+delta_t]
                elif i-delta_f < 0 and j-delta_t < 0:
                    box = DSP[0:i+delta_f,0:j+delta_t]
                elif i-delta_f >= 0 and i+delta_f <= stop_f and j-delta_t < 0:  
                    box = DSP[i-delta_f:i+delta_f,0:j+delta_t]
                elif i+delta_f > stop_f and j-delta_t >= 0 and j+delta_t <=stop_t:
                    box = DSP[i-delta_f:stop_f,j-delta_t:j+delta_t]   
                elif i+delta_f > stop_f and j+delta_t >stop_t:
                    box = DSP[i-delta_f:stop_f,j-delta_t:stop_t]   
                elif i-delta_f >= 0 and i+delta_f <= stop_f and j+delta_t >stop_t:  
                    box = DSP[i-delta_f:i+delta_f,j-delta_t:stop_t]
                elif i-delta_f >= 0 and i+delta_f > stop_f and j-delta_t < 0:
                    box = DSP[i-delta_f:stop_f,0:j+delta_t]
                elif i-delta_f < 0 and j+delta_t > stop_t:
                    box = DSP[0:i+delta_f,j-delta_t:stop_t]

                if box.size:
                    cv2box = cv2.resize(box.astype('float64'), (256,256), interpolation = cv2.INTER_CUBIC)
                    cv2box = (cv2box - amin) / (amax - amin)
                    cv2box = cv2box.reshape((-1, 256, 256, 1))

                    pred = model.predict(cv2box)                
                    result_array[i,j,r,0] = pred[0][0]
                    result_array[i,j,r,1] = pred[0][1]
                    result_array_full[i,i_rf[0]+j,r,1] = pred[0][1]


# In[17]:


fig, ax = plt.subplots(1)
ax.imshow(np.flipud(DSPxarr),cmap='jet')
#plt.rcParams['figure.figsize'] = [20, 10]
layer = 1 # 0 BG, 1 Burst
r = 0
color = 'r'
a = np.where(result_array_full[:,:,r,layer] > 0.75)
for idx in range(len(a[0])):
    rect = patches.Rectangle((a[1][idx]-ratio[r][1]/2,1020-a[0][idx]-ratio[r][0]/2),ratio[r][1],ratio[r][0],linewidth=0.5,fc=(1,0,0,0.5), ec='none')
    ax.add_patch(rect)
    

fig, ax = plt.subplots(1)
ax.imshow(np.flipud(DSPxarr),cmap='jet') 
r = 1
color = 'r'
a = np.where(result_array_full[:,:,r,layer] > 0.75)
for idx in range(len(a[0])):
    rect = patches.Rectangle((a[1][idx]-ratio[r][1]/2,1020-a[0][idx]-ratio[r][0]/2),ratio[r][1],ratio[r][0],linewidth=0.5,fc=(1,0,0,0.5), ec='none')
    ax.add_patch(rect)    
plt.show()


# In[18]:


'''
import numpy as np
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Input,Flatten,Activation
from keras.layers import GlobalMaxPooling2D
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm 


MODEL_NAME = 'UTR_CNN-V3-full_frame.model'
if os.path.exists('/home/stefan/Dokumente/MachineLearning/UTR2_Python/' + MODEL_NAME):
    model = load_model(MODEL_NAME)
    print('Loaded: ' + MODEL_NAME)

'''    
  
amax = np.amax(DSPxarr)
amin = np.amin(DSPxarr)

stop_f = DSPxarr.shape[0] # maximum frequency index 
stop_t = DSPxarr.shape[1] # maximum time index

ratio = [[500,50]]#,[500,100]]
classnum = 2
rationum = len(ratio)

#result_array = np.zeros((stop_f+1, stop_t+1,rationum,classnum))

result_array_full = np.zeros((stop_f+1, stop_t+1,rationum,classnum))
result_array_full_shallow = np.zeros((stop_f+1, stop_t+1,rationum,classnum))

for i_rf in tqdm(idx_range_fin):
#i_rf =  idx_range_fin[7]
#for i in range(1):

    DSP = DSPxarr[:,i_rf[0]:i_rf[1]]
    for r,ri in tqdm(enumerate(ratio)):   
        delta_f = round(ri[0]/2) # half window height
        delta_t = round(ri[1]/2) # half window width
       
        
        step_f = round(ri[0]/2)        
        step_f = 50
        
        f = np.arange(0, stop_f, step_f, dtype=None)
        
        step_t = round(ri[1])
        step_t = 50
        t = np.arange(0, stop_t, step_t, dtype=None)

        for i in f:
            for j in t:     
                if i-delta_f >= 0 and i+delta_f <= stop_f and j-delta_t >= 0 and j+delta_t <=stop_t:
                    box = DSP[i-delta_f:i+delta_f,j-delta_t:j+delta_t]            
                elif i-delta_f < 0 and j-delta_t >= 0 and j+delta_t <=stop_t:
                    box = DSP[0:i+delta_f,j-delta_t:j+delta_t]
                elif i-delta_f < 0 and j-delta_t < 0:
                    box = DSP[0:i+delta_f,0:j+delta_t]
                elif i-delta_f >= 0 and i+delta_f <= stop_f and j-delta_t < 0:  
                    box = DSP[i-delta_f:i+delta_f,0:j+delta_t]
                elif i+delta_f > stop_f and j-delta_t >= 0 and j+delta_t <=stop_t:
                    box = DSP[i-delta_f:stop_f,j-delta_t:j+delta_t]   
                elif i+delta_f > stop_f and j+delta_t >stop_t:
                    box = DSP[i-delta_f:stop_f,j-delta_t:stop_t]   
                elif i-delta_f >= 0 and i+delta_f <= stop_f and j+delta_t >stop_t:  
                    box = DSP[i-delta_f:i+delta_f,j-delta_t:stop_t]
                elif i-delta_f >= 0 and i+delta_f > stop_f and j-delta_t < 0:
                    box = DSP[i-delta_f:stop_f,0:j+delta_t]
                elif i-delta_f < 0 and j+delta_t > stop_t:
                    box = DSP[0:i+delta_f,j-delta_t:stop_t]
               
                
                if box.size:              
                    cv2box = cv2.resize(box.astype('float64'), (256,256), interpolation = cv2.INTER_CUBIC)
                    cv2box = (cv2box - amin) / (amax - amin)
                    cv2box = cv2box.reshape((-1, 256, 256, 1))

                    pred = model.predict(cv2box)

                    result_array_full[i-5:i+5,(i_rf[0]+j)-5:(i_rf[0]+j)+5,r,1] = pred[0][1]                                        
                    result_array_full_shallow[i,i_rf[0]+j,r,1] = pred[0][1]
print('Done')


# In[19]:


fig, ax = plt.subplots(1)
ax.imshow(np.flipud(DSPxarr),cmap='jet')
for i in idx_range_fin:  
    rect = patches.Rectangle((i[0],0),i[1]-i[0],1023,linewidth=1,edgecolor='white',facecolor='none')
    ax.add_patch(rect) 
plt.show()
layer = 1 # 0 BG, 1 Burst
r = 0
color = 'r'
fig, ax = plt.subplots(1)
ax.imshow(np.flipud(result_array_full[:,:,r,layer]),cmap='jet')
for i in idx_range_fin:  
    rect = patches.Rectangle((i[0],0),i[1]-i[0],1023,linewidth=1,edgecolor='white',facecolor='none')
    ax.add_patch(rect) 
plt.show()


# In[20]:


import copy
result_cp = copy.deepcopy(result_array_full[:,:,:,:])


# In[21]:


result_cp[result_cp[:,:,0,1] < 0.55] = 0


# In[22]:


fig, ax = plt.subplots(1)
ax.imshow(np.flipud(result_cp[:,:,r,layer]),cmap='jet')
for i in idx_range_fin:  
    rect = patches.Rectangle((i[0],0),i[1]-i[0],1023,linewidth=1,edgecolor='y',facecolor='none')
    ax.add_patch(rect) 
plt.show()


# In[ ]:





# In[23]:


result_array_full_zeros = np.zeros((result_array_full.shape))
result_array_full_zeros_shallow = np.zeros((result_array_full.shape))


# In[24]:


DSPxarr.shape


# In[25]:


result_array_full_zeros.shape


# In[26]:


result_array_full.shape


# In[27]:


result_array_full_zeros[845:855,:] = result_array_full[845:855,:]
result_array_full_zeros[595:605,:] = result_array_full[595:605,:]
result_array_full_zeros[395:405,:] = result_array_full[395:405,:]
result_array_full_zeros[245:255,:] = result_array_full[245:255,:]
result_array_full_zeros[145:155,:] = result_array_full[145:155,:]


# In[28]:


fig, ax = plt.subplots(1)
ax.pcolormesh(DSPxarr,cmap='jet')
for i in idx_range_fin:  
    rect = patches.Rectangle((i[0],0),i[1]-i[0],1023,linewidth=1,edgecolor='white',facecolor='none')
    ax.add_patch(rect) 
plt.show()


#fig, ax = plt.subplots(1)
#ax.pcolormesh(result_array_full[:,:,0,1],cmap='gray')
#plt.show()

fig, ax = plt.subplots(1)
ax.pcolormesh(result_array_full_zeros[:,:,0,1],cmap='jet')
#for i in idx_range_fin:  
#    rect = patches.Rectangle((i[0],0),i[1]-i[0],1024,linewidth=1,edgecolor='white',facecolor='none')
#    ax.add_patch(rect) 
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[29]:


ARR = result_array_full_shallow[:,:,0,1]
print(ARR.shape)


# In[30]:


from scipy.interpolate import griddata


# In[31]:


grid_x, grid_y = np.mgrid[0:1024:1, 0:3601:1]
points = []
for z in zip(ARR.nonzero()[0],ARR.nonzero()[1]):
    points.append(np.array(z))

values = [ARR[x] for x in zip(ARR.nonzero()[0],ARR.nonzero()[1])]

interp = griddata(points, values, (grid_x, grid_y), method='cubic')


# In[ ]:





# In[32]:


fig, ax = plt.subplots(1)
ax.imshow(np.flipud(interp),cmap='gray')
plt.show()


# In[33]:


interp_sep = np.zeros((interp.shape))
for i in idx_range_fin:
    interp_sep[:,i[0]:i[1]] = interp[:,i[0]:i[1]]


# In[34]:


fig, ax = plt.subplots()
plt.rcParams["figure.figsize"] = (30,10)
CS = ax.contour(grid_y, grid_x, np.flipud(interp_sep), levels = [0.85,0.90,0.95], linewidths = 3, colors = ['c','yellow','r'])
ax.clabel(CS, inline=1, fontsize=10)
ax.imshow(np.flipud(result_array_full[:,:,r,layer]),cmap='jet')
for i in idx_range_fin:  
    rect = patches.Rectangle((i[0],0),i[1]-i[0],1024,linewidth=3,edgecolor='gray',facecolor='none')
    ax.add_patch(rect) 

ax.set_xticks(np.arange(0,DSPxarr.shape[1],350))
ax.set_xticks(np.arange(0,DSPxarr.shape[1],50), minor=True)

r = ((29.8211-17.3333)/1023) # the ratio of how much a pixel is worth in MHz
z = [i - 17.3333 for i in [28,26,24,22,20,18]]
frq_idx = [zi / r for zi in z]
frq_idx = [xi + 1023 - 873.8155719982702 - 54 for xi in frq_idx]
frq_idx = np.round(frq_idx,4)

ax.set_yticks(frq_idx)
ax.set_yticks(frq_idx_minor, minor=True)

frq = [18,20,22,24,26,28]
ax.set_yticklabels(frq)
ax.set_ylabel('Frequency [MHz]')
ax.set_xticklabels([])
ax.set_xticklabels(tvec(X[1]['start_time'],start_IDX, end_IDX))
ax.set_xlabel('Time [HH:MM:SS]')

fname = '/home/stefan/Dokumente/EPSC_2019/Images/Final/Burst_Dotty_Contour.png'

plt.savefig(fname, dpi=400, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)



plt.show()    


plt.rcParams["figure.figsize"] = (30,10)

fig, ax = plt.subplots()
CS = ax.contour(grid_y, grid_x, np.flipud(interp_sep), levels = [0.85,0.90,0.95],linewidths = 3, colors = ['c','yellow','r'])
ax.clabel(CS, inline=1, fontsize=10)
ax.imshow(np.flipud(DSPxarr),cmap='jet')
for i in idx_range_fin:  
    rect = patches.Rectangle((i[0],0),i[1]-i[0],1024,linewidth=3,edgecolor='gray',facecolor='none')
    ax.add_patch(rect)
    
ax.set_xticks(np.arange(0,DSPxarr.shape[1],350))
ax.set_xticks(np.arange(0,DSPxarr.shape[1],50), minor=True)

r = ((29.8211-17.3333)/1023) # the ratio of how much a pixel is worth in MHz
z = [i - 17.3333 for i in [28,26,24,22,20,18]]
frq_idx = [zi / r for zi in z]
frq_idx = [xi + 1023 - 873.8155719982702 - 54 for xi in frq_idx]
frq_idx = np.round(frq_idx,4)

ax.set_yticks(frq_idx)
ax.set_yticks(frq_idx_minor, minor=True)

frq = [18,20,22,24,26,28]
ax.set_yticklabels(frq)
ax.set_ylabel('Frequency [MHz]')
ax.set_xticklabels([])
ax.set_xticklabels(tvec(X[1]['start_time'],start_IDX, end_IDX))
ax.set_xlabel('Time [HH:MM:SS]')

fname = '/home/stefan/Dokumente/EPSC_2019/Images/Final/Burst_Contour.png'

plt.savefig(fname, dpi=400, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)




plt.show()


# In[35]:


test_frequencies = [0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
test_frequencies = [0,100,200,300,400,500,600,700,800,900,1000]


# In[36]:


result_array_full_zeros_shallow = np.zeros((result_array_full.shape))
for tf in test_frequencies:
    if tf == 1000:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:,0,1] = result_array_full_shallow[tf,:]
    if tf == 950:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:,0,1] = result_array_full_shallow[tf,:]    
    if tf == 900:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:,0,1] = result_array_full_shallow[tf,:]
    if tf == 850:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:,0,1] = result_array_full_shallow[tf,:]
    if tf == 800:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]
    if tf == 750:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]
    if tf == 700:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]    
    if tf == 650:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]
    if tf == 600:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]
    if tf == 550:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]
    if tf == 500:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]
    if tf == 450:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]
    if tf == 400:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]
    if tf == 350:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]
    if tf == 300:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]
    if tf == 250:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]
    if tf == 200:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]
    if tf == 150:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]
    if tf == 100:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]
    if tf == 50:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]
    if tf == 0:
        result_array_full_zeros_shallow[tf,:,0,1] = interp_sep[tf,:]
        #result_array_full_zeros_shallow[tf,:] = result_array_full_shallow[tf,:]    


# In[37]:


all_driftlines = []
drift_results = []
duration_results = []

all_driftlines_results = []

offset2 = [25,200,600,1050,0,0,0,2250,0]

for bi, idx_bi in enumerate(idx_range_fin):
    offset = idx_bi[0]
    offset = offset2[bi]
    #bnine = interp_sep[:,idx_bi[0]:idx_bi[1]]
    bnine = result_array_full_zeros_shallow[:,idx_bi[0]:idx_bi[1],0,1]

    lvl = 0.88
    ranges = [np.where(bnine[0,:] > lvl),
              np.where(bnine[50,:] > lvl),
              np.where(bnine[100,:] > lvl),
              np.where(bnine[150,:] > lvl),
              np.where(bnine[200,:] > lvl),
              np.where(bnine[250,:] > lvl),
              np.where(bnine[300,:] > lvl),
              np.where(bnine[350,:] > lvl),
              np.where(bnine[400,:] > lvl),
              np.where(bnine[450,:] > lvl),
              np.where(bnine[500,:] > lvl),
              np.where(bnine[550,:] > lvl),
              np.where(bnine[600,:] > lvl),
              np.where(bnine[650,:] > lvl),
              np.where(bnine[700,:] > lvl),
              np.where(bnine[750,:] > lvl),
              np.where(bnine[800,:] > lvl),
              np.where(bnine[850,:] > lvl),
              np.where(bnine[900,:] > lvl),
              np.where(bnine[950,:] > lvl),
              np.where(bnine[1000,:] > lvl)]

    drift = []
    duration = []
    for i in ranges:
        if len(i[0]) > 0:  
            if i[0][0] == i[0][-1]:
                drift.append(i[0][0])
            else:
                drift.append(((i[0][-1]-i[0][0])/2))
            duration.append((i[0][-1]-i[0][0]))
    
    if drift:
        drift_results.append(drift)        
        duration_results.append(duration)   
        drift = [x+offset for x in drift]
        all_driftlines_results.append(drift)  

   


# In[38]:


min_good_steps = 5

colors = ['green','blue','orange','red','pink','yellow','cyan','magenta','purple']
plt.rcParams["figure.figsize"] = (15,10)
fig, ax = plt.subplots()
plt.rcParams["figure.figsize"] = (15,10)
CS = ax.contour(grid_y, grid_x, interp_sep, levels = [0.85],linewidths = 2, colors = ['c'])
#ax.clabel(CS, inline=1, fontsize=10)
f = test_frequencies
ax.pcolormesh(DSPxarr,cmap='gray')
for i in idx_range_fin:  
    rect = patches.Rectangle((i[0],0),i[1]-i[0],1023,linewidth=2,edgecolor='gray',facecolor='none')
    ax.add_patch(rect) 

    
#all_driftlines_results =    all_driftlines_results[::-1]
for ii, i in enumerate(all_driftlines_results):
    fd = f[0:len(i)] 
    if len(fd) > min_good_steps:
        ax.plot(i, fd, label=ii,linewidth = 2, marker = 'o', color = colors[ii])

plt.legend(loc='upper right')


ax.set_xticks(np.arange(0,DSPxarr.shape[1],350))
ax.set_xticks(np.arange(0,DSPxarr.shape[1],50), minor=True)

r = ((29.8211-17.3333)/1023) # the ratio of how much a pixel is worth in MHz
z = [i - 17.3333 for i in [28,26,24,22,20,18]]
frq_idx = [zi / r for zi in z]
frq_idx = np.round(frq_idx,4)

ax.set_yticks(frq_idx)
ax.set_yticks(frq_idx_minor, minor=True)

frq = [28,26,24,22,20,18]
ax.set_yticklabels(frq)
ax.set_ylabel('Frequency [MHz]')
ax.set_xticklabels([])
ax.set_xticklabels(tvec(X[1]['start_time'],start_IDX, end_IDX))
ax.set_xlabel('Time [HH:MM:SS]')


fname = '/home/stefan/Dokumente/EPSC_2019/Images/Final/0_rect_final.png'

plt.savefig(fname, dpi=400, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)
plt.show()  


# In[39]:


colors = ['green','blue','orange','red','pink','yellow','cyan','magenta','purple']
plt.rcParams["figure.figsize"] = (15,5)

plt.subplot(1, 2, 1)
ratio = ((29.8211-17.3333)/1023)
f = test_frequencies

f = [(i * ratio)+18 for i in f]
for gg, g in enumerate(drift_results):
    dt = np.diff(g)
    dt = np.true_divide(dt,10)   
    dt = np.abs(dt)  
    dt = np.negative(dt)
    df = np.diff(f)
    dfi = df[0:len(dt)]
    fi = f[0:len(dt)]
    if len(fi)>=min_good_steps:
        #print(dfi,dt)
        #print(dfi/dt)
        print(fi)
        plt.plot(fi, dfi/dt, label=gg,linewidth = 2, linestyle = '-',color = colors[gg])
    

plt.title('Driftrate')

plt.xlim(18, 30)
plt.xlabel('f [MHz]')
plt.ylim(-5, 0)
plt.ylabel('df/dt [MHz/s]')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
ratio = ((32-8)/1023) # the ratio of how much a pixel is worth in MHz
ratio = ((29.8211-17.3333)/1023)
f = test_frequencies

f = [(i * ratio)+18 for i in f]

for dd, d in enumerate(duration_results):
    fd = f[0:len(d)]
    d = np.true_divide(d, 10)
    if len(fd)>=min_good_steps+1:        
        plt.plot(fd, d, label = dd, linewidth = 2,linestyle = '-', color = colors[dd])

plt.title('Duration')
plt.xlim(18, 30)
plt.xlabel('f [MHz]')
plt.ylim(0, 60)
plt.ylabel('d [s]')

plt.legend(loc='upper right')
plt.grid(True)


fname = '/home/stefan/Dokumente/EPSC_2019/Images/Final/Drift_Duration_Box_2_final.png'

plt.savefig(fname, dpi=400, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)




plt.show()


# In[40]:


colors = ['green','blue','orange','red','pink','yellow','cyan','magenta','purple']
plt.rcParams["figure.figsize"] = (15,10)

plt.subplot(1, 2, 1)
ratio = ((29.8211-17.3333)/1023)
f = test_frequencies

f = [(i * ratio)+18 for i in f]
for gg, g in enumerate(drift_results):
    dt = np.diff(g)
    dt = np.true_divide(dt,10)   
    dt = np.abs(dt)  
    dt = np.negative(dt)
    df = np.diff(f)
    dfi = df[0:len(dt)]
    fi = f[0:len(dt)]
    if len(fi)>=min_good_steps:
        dfdt = dfi/dt
        idx = np.isfinite(fi) & np.isfinite(dfdt)
        fi = np.asarray(fi)
        z = np.polyfit(fi[idx], dfdt[idx],2)
        p = np.poly1d(z)
        xp = np.linspace(f[0], f[-1], 10)
        plt.plot(xp, p(xp), label=gg, marker = 'None', linestyle = '-',color = colors[gg])
        plt.plot(fi, dfdt,marker = '.', linestyle = 'None',color = colors[gg])
    

plt.title('Driftrate')

plt.xlim(18, 30)
plt.xlabel('f [MHz]')
plt.ylim(-5, 0)
plt.ylabel('df/dt [MHz/s]')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
ratio = ((32-8)/1023) # the ratio of how much a pixel is worth in MHz
ratio = ((29.8211-17.3333)/1023)
f = test_frequencies

f = [(i * ratio)+18 for i in f]

for dd, d in enumerate(duration_results):
    fd = f[0:len(d)]
    d = np.true_divide(d, 10)
    if len(fd)>=min_good_steps+1:    
        fd = np.asarray(fd)
        idx = np.isfinite(fd) & np.isfinite(d)
        fi = np.asarray(fd)
        z = np.polyfit(fd[idx], d[idx],1)
        p = np.poly1d(z)
        xp = np.linspace(f[0], f[-1], 10)
        plt.plot(xp, p(xp), label=dd, marker = 'None', linestyle = '-',color = colors[dd])
        plt.plot(fd, d, marker = '.', linestyle = 'None', color = colors[dd])

plt.title('Duration')
plt.xlim(18, 30)
plt.xlabel('f [MHz]')
plt.ylim(0, 60)
plt.ylabel('d [s]')

plt.legend(loc='upper right')
plt.grid(True)


fname = '/home/stefan/Dokumente/EPSC_2019/Images/Final/Drift_Duration_Box_2_final_no_lines.png'

plt.savefig(fname, dpi=400, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)




plt.show()


# In[41]:


colors = ['green','blue','orange','red','pink','yellow','cyan','magenta','purple']
plt.rcParams["figure.figsize"] = (10,10)

plt.subplot(2, 1, 1)
ratio = ((29.8211-17.3333)/1023)
f = test_frequencies

f = [(i * ratio)+18 for i in f]

for gg, g in enumerate(drift_results):
    fd = f[0:len(g)]
    g = np.true_divide(g, 10)
    if len(fd)> 2:
        g = np.gradient(g)  
        plt.plot(fd, g, label=gg,linewidth = 3, linestyle = '-',color = colors[gg])


plt.title('Driftrate')

plt.xlim(18, 30)
plt.xlabel('f [MHz]')
plt.ylim(-10, 10)
plt.ylabel('df/dt [MHz/s]')
plt.grid(True)
plt.legend(loc='upper right')


plt.subplot(2, 1, 2)
ratio = ((32-8)/1023) # the ratio of how much a pixel is worth in MHz
ratio = ((29.8211-17.3333)/1023)
f = test_frequencies

f = [(i * ratio)+18 for i in f]

for dd, d in enumerate(duration_results):
    fd = f[0:len(d)]
    d = np.true_divide(d, 10)
    if len(fd)> 2:
        plt.plot(fd, d, label = dd, linewidth = 3,linestyle = '-', color = colors[dd])
        
        
plt.title('Duration')
plt.xlim(18, 30)
plt.xlabel('f [MHz]')
plt.ylim(0, 60)
plt.ylabel('d [s]')

plt.legend(loc='upper right')
plt.grid(True)

#fname = '/home/stefan/Dokumente/EPSC_2019/Images/Final/Drift_Duration_Box_2.png'

#plt.savefig(fname, dpi=400, facecolor='w', edgecolor='w',
#        orientation='portrait', papertype=None, format=None,
#        transparent=False, bbox_inches='tight', pad_inches=0.1,
#        frameon=None, metadata=None)




plt.show()


# In[42]:


colors = ['green','blue','orange','red','pink','yellow','cyan','magenta','purple']
plt.rcParams["figure.figsize"] = (30,10)
plt.subplot(1, 2, 1)

ratio = ((32-8)/1023) # the ratio of how much a pixel is worth in MHz
ratio = ((29.8211-17.3333)/1023)
f = [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950]
f = test_frequencies

f = [(i * ratio)+18 for i in f]

for gg, g in enumerate(drift_results):
    fd = f[0:len(g)]
    g = np.true_divide(g, 10)
    if len(fd)> 2:
        g = np.gradient(g)  
        plt.plot(fd, g, label=gg, linewidth = 3, color = colors[gg])


plt.title('Driftrate')

plt.xlim(18, 30)
plt.xlabel('f [MHz]')
plt.ylim(-20, 20)
plt.ylabel('df/dt [MHz/s]')
plt.grid(True)
plt.legend(loc='upper right')


plt.subplot(1, 2, 2)
ratio = ((32-8)/1023) # the ratio of how much a pixel is worth in MHz
ratio = ((29.8211-17.3333)/1023)
f = test_frequencies

f = [(i * ratio)+18 for i in f]

for dd, d in enumerate(duration_results):
    fd = f[0:len(d)]
    d = np.true_divide(d, 10)
    if len(fd)> 2:
        plt.plot(fd, d, label = dd, linewidth = 3, color = colors[dd])

plt.title('Duration')
plt.xlim(18, 30)
plt.xlabel('f [MHz]')
plt.ylim(0, 60)
plt.ylabel('d [s]')

plt.legend(loc='upper right')
plt.grid(True)


fname = '/home/stefan/Dokumente/EPSC_2019/Images/Final/Drift_Duration_Box_1.png'

plt.savefig(fname, dpi=400, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)



plt.show()


# In[43]:


a = np.arange(54.616-(1*abs((np.gradient(frq_idx)/4)[0])),873.8156+(3*abs((np.gradient(frq_idx)/4)[0])),abs((np.gradient(frq_idx)/4)[0]))
frq_idx_minor = np.round(a,4)

r = ((29.8211-17.3333)/1023) # the ratio of how much a pixel is worth in MHz
z = [i - 17.3333 for i in [28,26,24,22,20,18]]
frq_idx = [zi / r for zi in z]
frq_idx = np.round(frq_idx,4)


# In[44]:


colors = ['green','blue','orange','red','pink','yellow','cyan','magenta','purple']
plt.rcParams["figure.figsize"] = (30,10)
fig, ax = plt.subplots()
plt.rcParams["figure.figsize"] = (30,10)
CS = ax.contour(grid_y, grid_x, interp_sep, levels = [0.85],linewidths = 3, colors = ['c'])
#ax.clabel(CS, inline=1, fontsize=10)
f = [150,400,600,800]
ax.pcolormesh(DSPxarr,cmap='gray')
for i in idx_range_fin:  
    rect = patches.Rectangle((i[0],0),i[1]-i[0],1023,linewidth=3,edgecolor='gray',facecolor='none')
    ax.add_patch(rect) 

    
#all_driftlines_results =    all_driftlines_results[::-1]
for ii, i in enumerate(all_driftlines_results):
    fd = f[0:len(i)] 
    if len(fd) > 2:
        ax.plot(i, fd, label=ii,linewidth = 3, marker = 'o', color = colors[ii])

plt.legend(loc='upper right')


ax.set_xticks(np.arange(0,DSPxarr.shape[1],350))
ax.set_xticks(np.arange(0,DSPxarr.shape[1],50), minor=True)

r = ((29.8211-17.3333)/1023) # the ratio of how much a pixel is worth in MHz
z = [i - 17.3333 for i in [28,26,24,22,20,18]]
frq_idx = [zi / r for zi in z]
frq_idx = np.round(frq_idx,4)

ax.set_yticks(frq_idx)
ax.set_yticks(frq_idx_minor, minor=True)

frq = [28,26,24,22,20,18]
ax.set_yticklabels(frq)
ax.set_ylabel('Frequency [MHz]')
ax.set_xticklabels([])
ax.set_xticklabels(tvec(X[1]['start_time'],start_IDX, end_IDX))
ax.set_xlabel('Time [HH:MM:SS]')


fname = '/home/stefan/Dokumente/EPSC_2019/Images/Final/0_rect_final.png'

plt.savefig(fname, dpi=400, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)
plt.show()  


# In[ ]:


colors = ['green','blue','orange','red','pink','yellow','cyan','magenta','purple']
plt.rcParams["figure.figsize"] = (20,10)
fig, ax = plt.subplots()
plt.rcParams["figure.figsize"] = (20,10)
CS = ax.contour(grid_y, grid_x, interp_sep, levels = [0.95],linewidths = 3, colors = ['r'])
#ax.clabel(CS, inline=1, fontsize=10)
f = [150,400,600,800]
ax.pcolormesh(DSPxarr,cmap='gray')
for i in idx_range_fin:  
    rect = patches.Rectangle((i[0],0),i[1]-i[0],1023,linewidth=3,edgecolor='gray',facecolor='none')
    ax.add_patch(rect) 

    
#all_driftlines_results =    all_driftlines_results[::-1]
for ii, i in enumerate(all_driftlines_results):
    fd = f[0:len(i)] 
    if len(fd) > 2:
        ax.plot(i, fd, label=ii,linewidth = 3, color = colors[ii])

plt.legend(loc='upper right')


ax.set_xticks(np.arange(0,DSPxarr.shape[1],350))
ax.set_xticks(np.arange(0,DSPxarr.shape[1],50), minor=True)

r = ((29.8211-17.3333)/1023) # the ratio of how much a pixel is worth in MHz
z = [i - 17.3333 for i in [28,26,24,22,20,18]]
frq_idx = [zi / r for zi in z]
#frq_idx = [xi + 1023 - 873.8155719982702 - 54 for xi in frq_idx]
frq_idx = np.round(frq_idx,4)

ax.set_yticks(frq_idx)
ax.set_yticks(frq_idx_minor, minor=True)

frq = [28,26,24,22,20,18]
ax.set_yticklabels(frq)
ax.set_ylabel('Frequency [MHz]')
ax.set_xticklabels([])
ax.set_xticklabels(tvec(X[1]['start_time']))
ax.set_xlabel('Time [HH:MM:SS]')


fname = '/home/stefan/Dokumente/EPSC_2019/Images/Final/0_rect_2.png'

plt.savefig(fname, dpi=400, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)


plt.show()  


# In[ ]:





# In[ ]:


tmp = np.zeros([2, 5], dtype=int)


# In[ ]:


fname


# In[ ]:





# In[ ]:




