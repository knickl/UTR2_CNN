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
from keras.layers import LSTM



from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras import initializers
from keras.optimizers import RMSprop



def normalize(data):
	mmrange = []
	for i in data:
		if i['DSP'].shape[1]:
			max = np.amax(i['DSP'])
			min = np.amin(i['DSP'])
			mmrange.append([min,max])
	min = np.amin(mmrange)
	max = np.amax(mmrange)
	normdata_array = []
	for normdata in data:
		normdata['DSP'] = (normdata['DSP'] - min) / (max - min)
		if normdata['Label'] == 'TRAIN':
			normdata['DSP_CUTTED'] = (normdata['DSP_CUTTED'] - min) / (max - min)
		normdata_array = np.append(normdata_array,normdata)  
	del data    
	return normdata_array

def get_TRAIN_data():
	fname = '/media/WDSSD/MachineLearning/UTR-2/DATA/compact_dataframe_all_1000.npy'
	#fname = '/media/WDSSD/MachineLearning/UTR-2/DATA/compact_dataframe.npy'
	X = np.load(fname)
	X = normalize(X)
	train = []
	label = []
	for i in X:
		if i['DSP'].shape[1]:
			if i['Label'] == 'BG':
				arr = cv2.resize(i['DSP'].astype('float64'), (120,120), interpolation = cv2.INTER_CUBIC)
				arr = arr.reshape((120*120, 1))
				train.append(arr)
				label.append([1,0,0])
			elif i['Label'] == '3':
				arr = cv2.resize(i['DSP'].astype('float64'), (120,120), interpolation = cv2.INTER_CUBIC)
				arr = arr.reshape((120*120, 1))
				train.append(arr)
				label.append([0,1,0])
			elif i['Label'] == 'B3':
				arr = cv2.resize(i['DSP'].astype('float64'), (120,120), interpolation = cv2.INTER_CUBIC)
				arr = arr.reshape((120*120, 1))
				train.append(arr)
				label.append([0,0,1])


	train = np.asarray(train)
	label = np.asarray(label)    		
	return train, label		

def RNNmodel():

	#model = Sequential()
	#model.add(SimpleRNN(hidden_units,
	#				batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),
	#				return_sequences=True,
	#				kernel_initializer=initializers.RandomNormal(stddev=0.001),
	#				recurrent_initializer=initializers.Identity(gain=1.0),
	#				activation='relu',
	#				input_shape=X_train.shape[1:],
	#				stateful = 'True'))
	model = Sequential()
	model.add(LSTM(100,
					input_shape=X_train.shape[1:],
					#batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),
					dropout=0.0,
					recurrent_dropout=0.0,
					stateful=False,
					kernel_initializer='random_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(3,activation='relu'))
	#model.add(Dense(1,activation='sigmoid'))
	rmsprop = RMSprop(lr=0.0001)
	model.compile(loss='categorical_crossentropy', 
					optimizer=rmsprop,
					metrics=['accuracy'])



	#model.add(Dense(num_classes))
	#model.add(Activation('softmax'))
	#model.add(Activation('sigmoid'))
	#rmsprop = RMSprop(lr=learning_rate)
	#model.compile(loss='categorical_crossentropy',
	#		  optimizer=rmsprop,
	#		  metrics=['accuracy'])

	model.summary()
	return model




MODEL_NAME = 'UTR_RNN-{}-{}.model'.format('V1','LSTM')
X, y = get_TRAIN_data()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1, train_size=0.75)

if os.path.exists('/home/stefan/Dokumente/MachineLearning/UTR2_Python/' + MODEL_NAME):
	model = load_model(MODEL_NAME)
	print('Loaded: ' + MODEL_NAME)
else:


	model = RNNmodel()
	model.fit(X_train, y_train,
						batch_size=20,			  
						epochs=5,
						verbose=1,
						validation_data=(X_valid, y_valid))
	print('Fitted: ' + MODEL_NAME)
	model.save(MODEL_NAME)




score = model.evaluate(X_valid, y_valid, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])





#X_test = np.load('/media/WDSSD/MachineLearning/UTR-2/testIMG.npy')



print(model.predict(X_test.reshape(1,6000,1)),batch_size=240)

#print(y_valid[3])