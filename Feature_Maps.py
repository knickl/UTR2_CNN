#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

from keras import backend as K


# In[2]:


MODEL_NAME = 'UTR_CNN-{}-{}.model'.format('V3','full_frame')

if os.path.exists('/home/stefan/Dokumente/MachineLearning/UTR2_Python/' + MODEL_NAME):
	model = load_model(MODEL_NAME)
	print('Loaded: ' + MODEL_NAME)	


	X_valid = np.load('/media/WDSSD/MachineLearning/UTR-2/UTR-2 2002_DATA/Data/X_valid.npy',allow_pickle=True)	
	y_valid = np.load('/media/WDSSD/MachineLearning/UTR-2/UTR-2 2002_DATA/Data/y_valid.npy',allow_pickle=True)

else:
	model=CNN()
	X, y = get_TRAIN_data()

	X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X, y, random_state=1, train_size=0.75)


	np.save('/media/WDSSD/MachineLearning/UTR-2/UTR-2 2002_DATA/Data/X_valid.npy',X_valid)
	np.save('/media/WDSSD/MachineLearning/UTR-2/UTR-2 2002_DATA/Data/y_valid.npy',y_valid)

	model.fit(X_train_cv, y_train_cv,
		batch_size=24,
		epochs=10,
		verbose=1,
		validation_data=(X_valid, y_valid),
		callbacks=callbacks)
	print('Fitted: ' + MODEL_NAME)
	model.save(MODEL_NAME)


# In[3]:


def layer_activation(idx):


	print(y_valid[idx])
	print(model.predict(X_valid[idx].reshape((-1, 256, 256, 1))))


	testIMG = np.flipud(X_valid[idx].reshape((256, 256)))
	plt.imshow(np.flipud(X_valid[idx].reshape((256, 256))), aspect = 'auto', cmap = 'gist_ncar')
	plt.show()



	layer_outputs = [layer.output for layer in model.layers[:12]]
	activation_model = Model(inputs = model.input, outputs = layer_outputs)
	activations = activation_model.predict(X_valid[idx].reshape((-1, 256, 256, 1)))

	layer_names = []
	for layer in model.layers[:12]:
		layer_names.append(layer.name)
	images_per_row = 16

	for layer_name, layer_activation in zip(layer_names, activations):
		if 'conv2d' in layer_name:
			n_features = layer_activation.shape[-1]

			size = layer_activation.shape[1]

			n_cols = n_features // images_per_row
			display_grid = np.zeros((size * n_cols, images_per_row * size))
			for col in range(n_cols):
				for row in range(images_per_row):

					channel_image = layer_activation[0,:,:, col * images_per_row + row]
					channel_image -= channel_image.mean()
					channel_image *= 64
					channel_image += 128
					channel_image = np.clip(channel_image, 0 , 255).astype('uint8')

					display_grid[col * size : (col +1 ) * size, row * size : (row +1 ) * size] = np.flipud(channel_image)

			scale = 1./ size
			plt.figure(figsize = (scale * display_grid.shape[1], scale * display_grid.shape[0]))           
			plt.title(layer_name)
			plt.grid(False)
			plt.imshow(display_grid, aspect = 'auto', cmap = 'gist_ncar')

			plt.show()
	return


# In[94]:


start_IDX = 4700
end_IDX = 5150


# In[95]:


fname = '/media/WDSSD/MachineLearning/UTR-2/UTR-2 2002_DATA/Data/test_dataframe.npy'
X = np.load(fname,allow_pickle=True)
DSP = X[1]['DSP'][:,start_IDX:end_IDX]


# In[130]:


DSP.shape


# In[96]:


a = np.arange(149.8-(3*abs((np.gradient(frq_idx)/4)[0])),969+(1*abs((np.gradient(frq_idx)/4)[0])),abs((np.gradient(frq_idx)/4)[0]))
frq_idx_minor = np.round(a,4)


# In[97]:


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


# In[114]:


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
print(X[1]['start_time'])
frq = [18,20,22,24,26,28]
ax.set_yticklabels(frq)
ax.set_ylabel('Frequency [MHz]')
ax.set_xticklabels([])
ax.set_xticklabels(tvec(X[1]['start_time'],start_IDX, end_IDX))
ax.set_xlabel('Time [HH:MM:SS]')
ax.set_title('Fri, 02 Aug 2002')
fname = '/home/stefan/Dokumente/EPSC_2019/Images/Final/Burst_extracted.png'

plt.savefig(fname, dpi=400, facecolor='w', edgecolor='w',
       orientation='portrait', papertype=None, format=None,
       transparent=False, bbox_inches='tight', pad_inches=0.1,
       frameon=None, metadata=None) 


plt.show()


# In[84]:


amax = np.amax(DSP)
amin = np.amin(DSP)
box = DSP
cv2box = cv2.resize(box.astype('float64'), (256,256), interpolation = cv2.INTER_CUBIC)
cv2box = (cv2box - amin) / (amax - amin)


# In[111]:


plt.rcParams["figure.figsize"] = (30,10)


# In[ ]:





# In[121]:





# In[122]:


plt.rcParams["figure.figsize"] = (30,10)

testIMG = np.flipud(cv2box.reshape((256, 256)))
plt.imshow(np.flipud(cv2box.reshape((256, 256))), aspect = 'auto', cmap = 'jet')
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
frame1.axes.get_xaxis().set_ticks([])
frame1.axes.get_yaxis().set_ticks([])
plt.title('Reshaped: (256,256,1)')


#fname = '/home/stefan/Dokumente/EPSC_2019/Images/Final/Burst_extracted_squished.png'

#plt.savefig(fname, dpi=400, facecolor='w', edgecolor='w',
#       orientation='portrait', papertype=None, format=None,
#       transparent=False, bbox_inches='tight', pad_inches=0.1,
#       frameon=None, metadata=None) 


plt.show()







layer_outputs = [layer.output for layer in model.layers[:12]]
activation_model = Model(inputs = model.input, outputs = layer_outputs)
activations = activation_model.predict(cv2box.reshape((-1, 256, 256, 1)))

layer_names = []
for layer in model.layers[:12]:
    layer_names.append(layer.name)
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    if 'conv2d' in layer_name:
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]

        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):

                channel_image = layer_activation[0,:,:, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0 , 255).astype('uint8')

                display_grid[col * size : (col +1 ) * size, row * size : (row +1 ) * size] = np.flipud(channel_image)

        scale = 1./ size
        plt.figure(figsize = (scale * display_grid.shape[1], scale * display_grid.shape[0]))           
        plt.title(layer_name)
        plt.grid(False)        
        plt.imshow(display_grid, aspect = 'auto', cmap = 'jet')
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        
        
        
        fname = '/home/stefan/Dokumente/EPSC_2019/Images/Final/feature_maps'
        fname = fname + layer_name
        plt.savefig(fname, dpi=400, facecolor='w', edgecolor='w',
           orientation='portrait', papertype=None, format=None,
           transparent=False, bbox_inches='tight', pad_inches=0.1,
           frameon=None, metadata=None) 

        
        plt.show()


# In[ ]:





# In[131]:


idx = 1


x = cv2box.reshape((-1, 256, 256, 1))
preds = model.predict(cv2box.reshape((-1, 256, 256, 1)))

class_idx = np.argmax(preds[0])
class_output = model.output[:, class_idx]
last_conv_layer  = model.get_layer('conv2d_1')

grads = K.gradients(class_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis = (0,1,2))

iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(64):
    conv_layer_output_value[:,:,i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis = 1)	

heatmap	= np.maximum(heatmap,0)
heatmap /= np.max(heatmap)

heatmap = cv2.resize(heatmap, (256,256))
heatmap = np.uint8(255*heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
heatmap = np.flipud(heatmap)

img = np.flipud(cv2box.reshape((256, 256)))
fig, ax = plt.subplots(1)
ax.imshow(img,cmap='jet')
ax.imshow(heatmap, interpolation='nearest', alpha = 0.5)
plt.show()


# In[136]:


cv2box = cv2box.reshape((-1, 256, 256, 1))
score = model.predict(cv2box)


# In[142]:


print('Background propability:', score[0][0])
print('Type III propability:', score[0][1])


# In[144]:


layer_name = 'conv2d_4'
filter_index = 0
layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:,:,:,filter_index])


# In[ ]:





# In[ ]:





# In[ ]:





# In[170]:


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std()+ 1e-5)
    x *= 0.1
    
    x += 0.5
    x = np.clip(x,0,1)
    x *= 255
    x = np.clip(x,0,255).astype('uint8')
    return x


# In[193]:


def generate_pattern(layer_name, filter_index, size,input_img_data):
    layers_ouput = model.get_layer(layer_name).output
    print(layers_ouput.shape)
    
    
    loss = K.mean(layer_output[:,:,:,filter_index])
    grads = K.gradients(loss,model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads)))+1e-5)
    iterate = K.function([model.input],[loss,grads])
    
    step = 1.
    for i in range(40):
        loss_value , grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    
    img = input_img_data[0]
    return deprocess_image(img)
    


# In[196]:


size = 64
margin = 5
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
size = 150
for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j*8),size,DSP)
        
        
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        
        results[horizontal_start: horizontal_end, vertical_start, vertical_end, : ] = filter_img


# In[ ]:





# In[ ]:




