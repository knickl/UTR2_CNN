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

from keras import backend as K

def CNN():
	#Choose Sequential Model
	model = Sequential()
	#Conv Layer 1
	model.add(Conv2D(128, kernel_size=(3, 3), strides=1, padding='valid',activation='relu', data_format="channels_last", input_shape=(240, 240, 1)))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
	model.add(Dropout(0.2))

	#Conv Layer 2
	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.2))

	#Conv Layer 3
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.2))

	#Conv Layer 4
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.2))

	#Flatten the data for upcoming dense layers
	model.add(Flatten())

	#Dense Layers
	model.add(Dense(512))
	model.add(Activation('relu'))
	#Dropout 1
	model.add(Dropout(0.2))

	#Dense Layer 2
	model.add(Dense(256))
	model.add(Activation('relu'))
	#Dropout 2
	model.add(Dropout(0.2))

	#Sigmoid Layer
	model.add(Dense(3))
	model.add(Activation('sigmoid'))

	optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='binary_crossentropy',
				  optimizer=optimizer,
				  metrics=['accuracy'])

	model.summary()
	return model

def get_callbacks(filepath, patience=2):
	es = EarlyStopping('val_loss', patience=patience, mode="min")
	msave = ModelCheckpoint(filepath, save_best_only=True)
	return [es, msave]
file_path = ".model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=5)
CNN()

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
	fname = '/media/WDSSD/MachineLearning/UTR-2/DATA/compact_dataframe.npy'
	X = np.load(fname)
	X = normalize(X)
	train = []
	label = []
	for i in X:
		if i['DSP'].shape[1]:
			if i['Label'] == 'BG':
				arr = cv2.resize(i['DSP'].astype('float64'), (240,240), interpolation = cv2.INTER_CUBIC)
				arr = arr.reshape((240, 240,1))
				train.append(arr)
				label.append([1,0,0])
			elif i['Label'] == '3':
				arr = cv2.resize(i['DSP'].astype('float64'), (240,240), interpolation = cv2.INTER_CUBIC)
				arr = arr.reshape((240, 240,1))
				train.append(arr)
				label.append([0,1,0])
			elif i['Label'] == 'B3':
				arr = cv2.resize(i['DSP'].astype('float64'), (240,240), interpolation = cv2.INTER_CUBIC)
				arr = arr.reshape((240, 240,1))
				train.append(arr)
				label.append([0,0,1])


	train = np.asarray(train)
	label = np.asarray(label)    		
	return train, label		

MODEL_NAME = 'UTR_CNN-{}-{}.model'.format('V1','conv_basic')

if os.path.exists('/home/stefan/Dokumente/MachineLearning/UTR2_Python/' + MODEL_NAME):
	model = load_model(MODEL_NAME)
	print('Loaded: ' + MODEL_NAME)


	X_valid = np.load('/media/WDSSD/MachineLearning/UTR-2/X_valid.npy')	
	y_valid = np.load('/media/WDSSD/MachineLearning/UTR-2/y_valid.npy')

else:
	model=CNN()
	X, y = get_TRAIN_data()
	X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X, y, random_state=1, train_size=0.75)
	model.fit(X_train_cv, y_train_cv,
		batch_size=24,
		epochs=10,
		verbose=1,
		validation_data=(X_valid, y_valid),
		callbacks=callbacks)
	print('Fitted: ' + MODEL_NAME)
	model.save(MODEL_NAME)

#gmodel.load_weights(filepath=file_path)
#score = model.evaluate(X_valid, y_valid, verbose=1)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])


def heatmap(idx):
	x = X_valid[idx].reshape((-1, 240, 240, 1))
	preds = model.predict(X_valid[idx].reshape((-1, 240, 240, 1)))

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

	heatmap = cv2.resize(heatmap, (240,240))
	heatmap = np.uint8(255*heatmap)
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	heatmap = np.flipud(heatmap)

	img = np.flipud(X_valid[idx].reshape((240, 240)))
	fig, ax = plt.subplots(1)
	ax.imshow(img,cmap='gray')
	ax.imshow(heatmap, interpolation='nearest', alpha = 0.5)
	plt.show()

	return

def layer_activation(idx):


	print(y_valid[idx])
	print(model.predict(X_valid[idx].reshape((-1, 240, 240, 1))))


	testIMG = np.flipud(X_valid[idx].reshape((240, 240)))
	plt.imshow(np.flipud(X_valid[idx].reshape((240, 240))), aspect = 'auto', cmap = 'gist_ncar')
	plt.show()



	layer_outputs = [layer.output for layer in model.layers[:12]]
	activation_model = Model(inputs = model.input, outputs = layer_outputs)
	activations = activation_model.predict(X_valid[idx].reshape((-1, 240, 240, 1)))

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


for idx in range(156):	

	heatmap(idx)




#predicted_test=model.predict_proba(X_test)
#print(predicted_test)


'''
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
'''