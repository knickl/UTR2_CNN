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




MODEL_NAME = 'UTR_CNN-V1-conv_basic.model'
if os.path.exists('/home/stefan/Dokumente/MachineLearning/UTR2_Python/' + MODEL_NAME):
	model = load_model(MODEL_NAME)
	print('Loaded: ' + MODEL_NAME)

fname = '/media/WDSSD/MachineLearning/UTR-2/DATA/test_dataframe.npy'
X = np.load(fname,allow_pickle=True)


DSP = X[9]['DSP']

amax = np.amax(DSP)
amin = np.amin(DSP)

step= 204
stop_f = DSP.shape[0] # maximum frequency index 
stop_t = DSP.shape[1] # maximum time index

f = np.arange(0, stop_f+step, step, dtype=None)
t = np.arange(0, stop_t+step, step, dtype=None)

result_array = np.zeros((stop_f+1, stop_t+1,3))

ratio = [[100,100],[200,200],[400,400],[241,100],[482,200],[964,400],[100,241],[200,482],[400,964]]

for r in tqdm(ratio):   
	delta_f = round(r[0]/2) # half window height
	delta_t = round(r[1]/2) # half window width
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
				cv2box = cv2.resize(box.astype('float64'), (240,240), interpolation = cv2.INTER_CUBIC)
				cv2box = (cv2box - amin) / (amax - amin)
				cv2box = cv2box.reshape((-1, 240, 240, 1))

				pred = model.predict(cv2box)
				result_array[i-delta_t:i+delta_t,j-delta_f:j+delta_f,0] = result_array[i-delta_t:i+delta_t,j-delta_f:j+delta_f,0] + pred[0][0]
				result_array[i-delta_t:i+delta_t,j-delta_f:j+delta_f,1] = result_array[i-delta_t:i+delta_t,j-delta_f:j+delta_f,1] + pred[0][1]
				result_array[i-delta_t:i+delta_t,j-delta_f:j+delta_f,2] = result_array[i-delta_t:i+delta_t,j-delta_f:j+delta_f,2] + pred[0][2]


