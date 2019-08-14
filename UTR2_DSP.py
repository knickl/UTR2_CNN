import numpy as np
import matplotlib.pyplot as plt
import cv2
from sys import getsizeof
import os
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
import io
import sys
import pandas as pd
import pickle as pl
import copy


# --- --- --- --- --- --- --- #
# --- --- --- --- --- --- --- #
# --- --- --- --- --- --- --- #
# --- --- DEFINE  --- --- --- #

def normalize(array):
	# A Function that Normalizes an Array by simple Min/Max Normalization
	# Inputs:
	# 'array'	the input Array
	# Outputs:
	# 'array'	the Normalized output Array
	array = (array - np.amin(array)) / (np.amax(array) - np.amin(array)) #Simple Min/Max Normalization. Used to equalize the data for ML
	return	array

def Total_TimeUNIX(DynamicSpectrum,Total_Seconds):
	# A function that calculates start-stop time for the measurement
	# Inputs:
	# Outputs:
	dtfmt='%d.%m.%Y %H.%M.%S' # timeformat
	Total_Time = datetime.utcfromtimestamp(int(DynamicSpectrum['start_time'])).strftime(dtfmt),(datetime.utcfromtimestamp(int(DynamicSpectrum['start_time'])) + timedelta(seconds=Total_Seconds)).strftime(dtfmt)
	return Total_Time 

def Total_Time_old(DynamicSpectrum):
	# A Function to define a datetime.dateime Object for Start and Stop of the Measurement
	# Inputs:
	# Output:
	Total_Seconds = DynamicSpectrum['TNS']*(DynamicSpectrum['millisec']/1000) #total time of a sample in seconds
	Total_Time = Total_TimeUNIX(DynamicSpectrum,Total_Seconds)
	DynamicSpectrum['Total_Time'] =  Total_Time

	return DynamicSpectrum


def Total_Time(DynamicSpectrum):
	# A Function to define a datetime.dateime Object for Start and Stop of the Measurement
	# Inputs:
	# Output:
	TNS = int(DynamicSpectrum['TNS'])
	start_time = datetime.utcfromtimestamp(int(DynamicSpectrum['start_time']))
	end_time = start_time + timedelta(seconds=TNS/10)
	dtfmt='%d.%m.%Y %H.%M.%S' # timeformat
	Total_Time = (start_time.strftime(dtfmt),end_time.strftime(dtfmt))	
	DynamicSpectrum['Total_Time'] =  Total_Time
	return DynamicSpectrum



def Time_IDX(DynamicSpectrum,pad):
	# A Function that calculates the index in the Dynamic Spectrum when a Burst happened
	# Inputs:
	# 'time'		A touple with the String-Time Object when the Measurement was taken
	# 'BurstTime'	A touple with the String-Time Object when the Burst happened
	# Outputs:
	# A Touple with the following Elements:
	# 'idx0'		First index of the Burst in the Array - (!) not a Timestamp
	# 'idx'			Last index of the Burst in the Array - (!) not a Timestamp
	dtfmt='%d.%m.%Y %H.%M.%S' # timeformat

	if pad == True:
		idx0 = ((datetime.strptime(DynamicSpectrum['BurstTime_Padded'][0], dtfmt)-datetime.strptime(DynamicSpectrum['Total_Time'][0], dtfmt)).total_seconds()*(1000/DynamicSpectrum['millisec'])).astype(np.int64) # (the seconds after start).(transformed into element after start).(as an integer)
		idx1 = ((datetime.strptime(DynamicSpectrum['BurstTime_Padded'][1], dtfmt)-datetime.strptime(DynamicSpectrum['Total_Time'][0], dtfmt)).total_seconds()*(1000/DynamicSpectrum['millisec'])).astype(np.int64) # (the seconds after end).(transformed into element after start).(as an integer)
	elif pad == False:
		idx0 = ((datetime.strptime(DynamicSpectrum['BurstTime'][0], dtfmt)-datetime.strptime(DynamicSpectrum['Total_Time'][0], dtfmt)).total_seconds()*(1000/DynamicSpectrum['millisec'])).astype(np.int64) # (the seconds after start).(transformed into element after start).(as an integer)
		idx1 = ((datetime.strptime(DynamicSpectrum['BurstTime'][1], dtfmt)-datetime.strptime(DynamicSpectrum['Total_Time'][0], dtfmt)).total_seconds()*(1000/DynamicSpectrum['millisec'])).astype(np.int64) # (the seconds after end).(transformed into element after start).(as an integer)
	return idx0,idx1

def Test_IfIsInTime(DynamicSpectrum,BurstTime):
	# A Function that tests if a Burst is within the Timewindow of the Measurement
	# START MEASUREMENT < START BURST and END BURST < END MEASUREMENT 
	# Inputs:
	# Outputs:
	# 'Boolean'		Returns a boolean Statement if or if not the Burst is within the Timewindow

	dtfmt='%d.%m.%Y %H.%M.%S' # timeformat
	return bool(datetime.strptime(DynamicSpectrum['Total_Time'][0], dtfmt) < datetime.strptime(BurstTime[0], dtfmt)  and datetime.strptime(BurstTime[1], dtfmt) < datetime.strptime(DynamicSpectrum['Total_Time'][1], dtfmt) )

def Pad_Time(DynamicSpectrum,seconds):
	# A Function that pads the TimeTouple of the burst with a few extra seconds to increase the extracted feature
	# Inputs:
	# 'DynamicSpectrum'
	# 'seconds'				A Touple of seconds: padding before and after end
	# Outputs:
	dtfmt='%d.%m.%Y %H.%M.%S' # timeformat	
	BurstTime_Padded = (datetime.strptime(DynamicSpectrum['BurstTime'][0], dtfmt) - timedelta(0,seconds[0])).strftime(dtfmt),(datetime.strptime(DynamicSpectrum['BurstTime'][1], dtfmt) + timedelta(0,seconds[1])).strftime(dtfmt)	
	DynamicSpectrum['BurstTime_Padded'] =  BurstTime_Padded
	return DynamicSpectrum

def DynamicSpectrum_cutter(DynamicSpectrum,idx):
	# A Function that cuts the Feature out of the Array
	# Inputs:
	# 'DynamicSpectrum'			A Dictionary with lot of info
	# Outputs:
	# 'DynamicSpectrum'			A Dictionary with lot of info and a cutted DSP


	DSP = DynamicSpectrum['DSP'][:,idx[0]:idx[1]]

	if not DSP.shape[1] == 0:
		DynamicSpectrum['DSP_CUTTED'] = DSP
	else:
		DSP = DynamicSpectrum['DSP'][:,int(idx[0]/10):int(idx[1]/10)]
		DynamicSpectrum['DSP_CUTTED'] = DSP

	return DynamicSpectrum


def save(fname,path,source_path):
	#fn = os.path.basename(fname)
	#name = os.path.splitext(fn)[0]

	DynamicSpectrum = Load_DSP_File(fname) # get the complete Array and return a list of valuable elements
	DynamicSpectrum = Fetch_DSP_Data(fname,DynamicSpectrum)
	DynamicSpectrum = Total_Time(DynamicSpectrum)

	x = False
	Burst_Parameters = timestampsIO(source_path + '/timestamps.ts')[1:]

	for BParams in Burst_Parameters:

		BurstTime = BParams['BurstTime']
		# Test if Burst is in Measurement Timewindow
		if Test_IfIsInTime(DynamicSpectrum,BurstTime):
			destination_path = '/media/WDSSD/MachineLearning/UTR-2/TRAIN'		
			DynamicSpectrum.update(BParams)

			#DynamicSpectrum.update({'Label' : 'TRAIN'})
			DynamicSpectrum['Label'] = 'TRAIN'
			Prepsave(DynamicSpectrum,fname,path,source_path,destination_path)
			x = True	

	if not x:
		destination_path = '/media/WDSSD/MachineLearning/UTR-2/TEST'
		#DynamicSpectrum.update({'Label' : 'TEST'})
		DynamicSpectrum['Label'] = 'TEST'
		#Prepsave(DynamicSpectrum,fname,path,source_path,destination_path)

	return	

def DSP_IMAGE(fname,path,source_path):

	DynamicSpectrum = Load_DSP_File(fname) # get the complete Array and return a list of valuable elements
	DynamicSpectrum = Fetch_DSP_Data(fname,DynamicSpectrum)
	DynamicSpectrum = Total_Time(DynamicSpectrum)
	DynamicSpectrum['Label'] = 'TEST'

	
	print(DynamicSpectrum['start_time'])
	title = DynamicSpectrum['Total_Time'][0] + ' until '+ DynamicSpectrum['Total_Time'][1] + ' [UT]'	
	fig,ax = plt.subplots()	
	ax.pcolormesh(DynamicSpectrum['DSP'],cmap='gist_ncar')
	ax.set_yticklabels([])
	ax.set_yticklabels([8., 13., 18., 23., 28., 33.])
	ax.set_ylabel('Frequency [kHz]')		
	timevec = TimeVector(DynamicSpectrum,'TEST')
	ax.set_xticklabels([])
	#ax.axis('tight')
	#ax.axis('off')
	#fig.subplots_adjust(top=1.0)
	#fig.subplots_adjust(bottom=0.0)
	#fig.subplots_adjust(left=0.0)
	#fig.subplots_adjust(right=1.0)
	ax.set_xticklabels(timevec)
	ax.set_ylabel('Time [HH:MM:SS]')
	ax.set_title(title)

	plt.show()

	


	return




def getDSP_DATAFRAME(fname,path,source_path):
	DynamicSpectrum = Load_DSP_File(fname) # get the complete Array and return a list of valuable elements
	DynamicSpectrum = Fetch_DSP_Data(fname,DynamicSpectrum)
	DynamicSpectrum = Total_Time(DynamicSpectrum)
	DSP_DATAFRAME = []
	Burst_Parameters = timestampsIO(source_path + '/timestamps.ts')[1:]

	# Create a temporary DSP Array that can be resized for every iteration if a Burst is found in the spectrum.
	# Delte it afterwards
	DSP_TMP = np.copy(DynamicSpectrum['DSP'])
	x = False
	for BParams in Burst_Parameters:

		BurstTime = BParams['BurstTime']
		# Test if Burst is in Measurement Timewindow
		if Test_IfIsInTime(DynamicSpectrum,BurstTime):
			#Add Burst Parameters to current dynamic Spectrum
			DynamicSpectrum = {**DynamicSpectrum, **BParams}
			DynamicSpectrum['DSP'] = DSP_TMP

			DynamicSpectrum['Label'] = 'TRAIN'
			Pad_Time(DynamicSpectrum,(0,0))
			if Test_IfIsInTime(DynamicSpectrum,DynamicSpectrum['BurstTime_Padded']):
				DynamicSpectrum = DynamicSpectrum_cutter(DynamicSpectrum,Time_IDX(DynamicSpectrum,True)) #cut out the burst from the current array
			else:
				DynamicSpectrum = DynamicSpectrum_cutter(DynamicSpectrum,Time_IDX(DynamicSpectrum,False)) #cut out the burst from the current array
			
			if DynamicSpectrum['DSP'].shape[1]:
				if DynamicSpectrum['DSP'].shape[1] > 240 :
					resizeDSP = cv2.resize(DynamicSpectrum['DSP'].astype('float64'), (640,240), interpolation = cv2.INTER_CUBIC)
					DynamicSpectrum['DSP'] = resizeDSP	

			if DynamicSpectrum['DSP_CUTTED'].shape[1]:
				if DynamicSpectrum['DSP_CUTTED'].shape[1] > 5:
					resizeDSP = cv2.resize(DynamicSpectrum['DSP_CUTTED'].astype('float64'), (240,240), interpolation = cv2.INTER_CUBIC)
					DynamicSpectrum['DSP_CUTTED'] = resizeDSP

			x = True	
			DSP_DATAFRAME = np.append(DSP_DATAFRAME,DynamicSpectrum)	
	del DSP_TMP

	if not x:
		DynamicSpectrum['Label'] = 'TEST'
		if DynamicSpectrum['DSP'].shape[1]:
			if DynamicSpectrum['DSP'].shape[1] > 240 :
				resizeDSP = cv2.resize(DynamicSpectrum['DSP'].astype('float64'), (640,240), interpolation = cv2.INTER_CUBIC)
				DynamicSpectrum['DSP'] = resizeDSP
				DSP_DATAFRAME = np.append(DSP_DATAFRAME,DynamicSpectrum)

	fn = os.path.basename(fname)
	name = os.path.splitext(fn)[0]		
	msg = f"'{name}': Framed"
	print(msg, end='\n')


	return DSP_DATAFRAME



def get_compact_DATAFRAME(fname,path,source_path,stamps):
	DynamicSpectrum = Load_DSP_File(fname) # get the complete Array and return a list of valuable elements
	DynamicSpectrum = Fetch_DSP_Data(fname,DynamicSpectrum)
	DynamicSpectrum = Total_Time(DynamicSpectrum)



	DSP_DATAFRAME = []
	DSP_TMP = np.copy(DynamicSpectrum['DSP'])


	for tstamps in stamps:
		if DynamicSpectrum['start_time'] == tstamps[0]:

			DynamicSpectrum['DSP'] = DSP_TMP
			DynamicSpectrum['DSP'] = DynamicSpectrum['DSP'][:,tstamps[2]:tstamps[3]]
			DynamicSpectrum['Label'] = tstamps[1]
			DynamicSpectrum['idx'] = [tstamps[2],tstamps[3]]
			#DSP_DATAFRAME = np.append(DSP_DATAFRAME,DynamicSpectrum)
			tmp = 	copy.deepcopy(DynamicSpectrum)
			DSP_DATAFRAME.append(tmp)
			del tmp
	del DSP_TMP	


	fn = os.path.basename(fname)
	name = os.path.splitext(fn)[0]		
	msg = f"'{name}': Framed"
	print(msg, end='\n')


	return DSP_DATAFRAME





def getDSP_BrightIII(fname,path,source_path):
	DynamicSpectrum = Load_DSP_File(fname) # get the complete Array and return a list of valuable elements
	DynamicSpectrum = Fetch_DSP_Data(fname,DynamicSpectrum)
	DynamicSpectrum = Total_Time(DynamicSpectrum)
	DSP_DATAFRAME = []
	Burst_Parameters = timestampsIO(source_path + '/timestamps.ts')[1:]

	for BParams in Burst_Parameters:

		BurstTime = BParams['BurstTime']
		# Test if Burst is in Measurement Timewindow
		if Test_IfIsInTime(DynamicSpectrum,BurstTime):
			#Add Burst Parameters to current dynamic Spectrum
			DynamicSpectrum = {**DynamicSpectrum, **BParams}

			DynamicSpectrum['Label'] = 'TEST'

			print(DynamicSpectrum['start_time'])
			title = DynamicSpectrum['Total_Time'][0] + ' --- '+ DynamicSpectrum['Total_Time'][1] + ' \n' + DynamicSpectrum['BurstTime'][0] + ' --- ' +DynamicSpectrum['BurstTime'][1]
			fig,ax = plt.subplots()	
			ax.pcolormesh(DynamicSpectrum['DSP'],cmap='gist_ncar')
			#ax.set_yticklabels([])
			#ax.set_yticklabels([8., 13., 18., 23., 28., 33.])
			#ax.set_ylabel('Frequency [kHz]')
				
			#timevec = TimeVector(DynamicSpectrum,'TEST')
			#ax.set_xticklabels([])
			#ax.axis('tight')
			#ax.axis('off')
			#fig.subplots_adjust(top=1.0)
			#fig.subplots_adjust(bottom=0.0)
			#fig.subplots_adjust(left=0.0)
			#fig.subplots_adjust(right=1.0)
			#ax.set_xticklabels(timevec)
			#ax.set_ylabel('Time [HH:MM:SS]')
			ax.set_title(title)

			plt.show()


	return 


def Prepsave(DynamicSpectrum,fname,path,source_path,destination_path):
	# A Function that prepares the actual Saving operation:
	# Inputs:
	# 'DynamicSpectrum'			A Dictionary Holding much information	
	# 'fname'
	# 'path'
	# 'source_path'
	# 'destination_path'
	# Outputs:
	# NONE

	fn = os.path.basename(fname)
	name = os.path.splitext(fn)[0]		

	if DynamicSpectrum['Label'] == 'TRAIN':
		Pad_Time(DynamicSpectrum,(0,0))
		if Test_IfIsInTime(DynamicSpectrum,DynamicSpectrum['BurstTime_Padded']):
			DynamicSpectrum = DynamicSpectrum_cutter(DynamicSpectrum,Time_IDX(DynamicSpectrum,True)) #cut out the burst from the current array
		else:
			DynamicSpectrum = DynamicSpectrum_cutter(DynamicSpectrum,Time_IDX(DynamicSpectrum,False)) #cut out the burst from the current array

	sdir = path.replace(source_path,'') # the new Sub-directory 
	save_File(DynamicSpectrum,name,destination_path,sdir)
	#plotDSP(DynamicSpectrum)

	return


def saveTEST(DynamicSpectrum,fname,path,source_path,destination_path,resize):
	# A Function that prepares the actual Saving operation:
	# Inputs:
	# 'DynamicSpectrum'			A Dictionary Holding much information	
	# 'fname'
	# 'path'
	# 'source_path'
	# 'destination_path'
	# Outputs:
	# NONE

	fn = os.path.basename(fname)
	name = os.path.splitext(fn)[0]
	sdir = path.replace(source_path,'') # the new Sub-directory 
	save_File(DynamicSpectrum,name,destination_path,sdir,resize)
	#plotDSP(DynamicSpectrum)

	return

def saveTRAIN(DynamicSpectrum,fname,path,source_path,destination_path,resize):
	# A Function that prepares the actual Saving operation:
	# Inputs:
	# 'DynamicSpectrum'
	# 'fname'
	# 'path'		
	# 'source_path'
	# 'destination_path'
	# Outputs:
	# NONE
	fn = os.path.basename(fname)
	name = os.path.splitext(fn)[0]	
	Pad_Time(DynamicSpectrum,(0,0))
	if Test_IfIsInTime(DynamicSpectrum,DynamicSpectrum['BurstTime_Padded']):
		DynamicSpectrum = DynamicSpectrum_cutter(DynamicSpectrum,Time_IDX(DynamicSpectrum,True)) #cut out the burst from the current array
	else:
		DynamicSpectrum = DynamicSpectrum_cutter(DynamicSpectrum,Time_IDX(DynamicSpectrum,False)) #cut out the burst from the current array
	sdir = path.replace(source_path,'') # the new Sub-directory 

	save_File(DynamicSpectrum,name,destination_path,sdir,resize)
	#plotDSP(DynamicSpectrum)
	return

def save_File(DynamicSpectrum,name,destination_path,sdir):
	# A Function that actually saves the Dynamic Spectrum in a Specific Path
	# Inputs:
	# 'DynamicSpectrum'
	# 'name'
	# 'destination_path'
	# 'sdir'
	# Outputs:
	# NONE

	ffname = destination_path + sdir + '/' + name #New Full Filename
	
	# Resize DSP and DSP_CUTTED Array to a 10th of the size:	
	if DynamicSpectrum['DSP'].shape[1]:
		if DynamicSpectrum['DSP'].shape[1] > 100 :
			#ssize = round(DynamicSpectrum['DSP'].shape[1] / 10)
			resizeDSP = cv2.resize(DynamicSpectrum['DSP'].astype('float64'), (250,50), interpolation = cv2.INTER_CUBIC)
			DynamicSpectrum['DSP'] = resizeDSP
			
	if DynamicSpectrum['Label'] == 'TRAIN':
		if DynamicSpectrum['DSP_CUTTED'].shape[1]:
			if DynamicSpectrum['DSP_CUTTED'].shape[1] > 50:
				#ssize = round(DynamicSpectrum['DSP_CUTTED'].shape[1] / 5)
				resizeDSP = cv2.resize(DynamicSpectrum['DSP_CUTTED'].astype('float64'), (25,50), interpolation = cv2.INTER_CUBIC)
				DynamicSpectrum['DSP_CUTTED'] = resizeDSP
		
	# Save individual Files	
	os.makedirs(os.path.dirname(ffname), exist_ok=True) # create the directory if it does not exist already

	# Save in one large List (Dataframe)	
	dfname = '/media/WDSSD/MachineLearning/UTR-2/DATA/dataframe.npy'

	
	if not os.path.isfile(dfname):		
		np.save(dfname,DynamicSpectrum)		
		msg = f"'{name}': Saved"
	else:
		dataframe = np.load(dfname)
		lst = np.append(dataframe, DynamicSpectrum)
		np.save(dfname,lst)		
		
		msg = f"'{name}': Saved"
	print(msg, end='\n')
	

	if DynamicSpectrum['Label'] == 'TRAIN':
		text = ffname + '.npy' + '\n' + DynamicSpectrum['date'] + '\n' + DynamicSpectrum['time'] + '\n'
		f= open('/media/WDSSD/MachineLearning/UTR-2/savelog.txt','a+')
		f.write(text)
		f.close() 
	return

def Create_DSP_NPY_data(source_path):
	# A Function that loads any .dsp Data from a given source Path and converts them into .npy Arrays for:
	# Training: 	If a Burst in the .dsp File according the the timestamp.ts File
	# Testing:		If not Burst is in it, so it can be used to Evaluate or Test the Data
	# Inputs:
	# 'source_path'		The Path Parent Path to the .dsp Files
	# Outputs:
	# NONE			It just stores files on the HD.
	for path, subdir, files in os.walk(source_path):
		if not os.path.basename(os.path.normpath(path)) == 'Calibration':
			if files:
				#saveFolder(files,path,source_path)

				# Continue here If you Like to Test each Individual Sub-File
				for name in files:
					fname =  os.path.join(path, name)
					ext = os.path.splitext(fname)[1]
					if ext == '.dsp':
						save(fname,path,source_path)

	return

def Create_DSP_IMAGE(source_path):
	# A Function that loads any .dsp Data from a given source Path and converts them into .npy Arrays for:
	# Training: 	If a Burst in the .dsp File according the the timestamp.ts File
	# Testing:		If not Burst is in it, so it can be used to Evaluate or Test the Data
	# Inputs:
	# 'source_path'		The Path Parent Path to the .dsp Files
	# Outputs:
	# NONE			It just stores files on the HD.
	for path, subdir, files in os.walk(source_path):
		if not os.path.basename(os.path.normpath(path)) == 'Calibration':
			if files:
				#saveFolder(files,path,source_path)

				# Continue here If you Like to Test each Individual Sub-File
				for name in files:
					fname =  os.path.join(path, name)
					ext = os.path.splitext(fname)[1]
					if ext == '.dsp':
						DSP_IMAGE(fname,path,source_path)

	return


def Create_DATAFRAME(source_path):
	# A Function that loads any .dsp Data from a given source Path and converts them into .npy Arrays for:
	# Training: 	If a Burst in the .dsp File according the the timestamp.ts File
	# Testing:		If not Burst is in it, so it can be used to Evaluate or Test the Data
	# Inputs:
	# 'source_path'		The Path Parent Path to the .dsp Files
	# Outputs:
	# NONE			It just stores files on the HD.
	dataframe = []
	DSP_DATAFRAME = []
	dfname = '/media/WDSSD/MachineLearning/UTR-2/DATA/compact_dataframe.npy'

	if os.path.isfile(dfname):		
		os.remove(dfname)
	fname = '/media/WDSSD/MachineLearning/UTR-2/UTR-2 2002/stamps.ts'
	with open(fname) as f:
		stamps = f.readlines()
	stamps = [x.strip().split() for x in stamps]  	
	lst = []
	for i in stamps:
		l = [int(i[0]), i[1], int(i[2]), int(i[3])]
		lst.append(l)
	stamps = lst	

	idx = 0
	for path, subdir, files in os.walk(source_path):

		if not os.path.basename(os.path.normpath(path)) == 'Calibration':
			if files:
				#saveFolder(files,path,source_path)

				# Continue here If you Like to Test each Individual Sub-File
				for name in files:
					fname =  os.path.join(path, name)
					ext = os.path.splitext(fname)[1]
					if ext == '.dsp':
						DSP_DATAFRAME = get_compact_DATAFRAME(fname,path,source_path,stamps)
						dataframe = np.append(dataframe,DSP_DATAFRAME)
						if len(dataframe) > 100:
							if not os.path.isfile(dfname):											
								np.save(dfname,dataframe)
								print('Saved new Dataframe')
							else: 							
								fromfile = np.load(dfname)
								lst = np.append(fromfile, dataframe)
								np.save(dfname,lst)	
								print('Saved appending Dataframe')
							dataframe = []
	if len(dataframe): #if is not empty
		if os.path.isfile(dfname):
			fromfile = np.load(dfname)
			lst = np.append(fromfile, dataframe)
			np.save(dfname,lst)	
		else:
			np.save(dfname,dataframe)
		print('Saved last appending Dataframe')					
							
	return



def Create_DSP_DATAFRAME(source_path):
	# A Function that loads any .dsp Data from a given source Path and converts them into .npy Arrays for:
	# Training: 	If a Burst in the .dsp File according the the timestamp.ts File
	# Testing:		If not Burst is in it, so it can be used to Evaluate or Test the Data
	# Inputs:
	# 'source_path'		The Path Parent Path to the .dsp Files
	# Outputs:
	# NONE			It just stores files on the HD.
	dataframe = []
	DSP_DATAFRAME = []
	dfname = '/media/WDSSD/MachineLearning/UTR-2/DATA/dataframe.npy'

	if os.path.isfile(dfname):		
		os.remove(dfname)

	idx = 0
	for path, subdir, files in os.walk(source_path):

		if not os.path.basename(os.path.normpath(path)) == 'Calibration':
			if files:
				#saveFolder(files,path,source_path)

				# Continue here If you Like to Test each Individual Sub-File
				for name in files:
					fname =  os.path.join(path, name)
					ext = os.path.splitext(fname)[1]
					if ext == '.dsp':
						DSP_DATAFRAME = getDSP_DATAFRAME(fname,path,source_path)
						dataframe = np.append(dataframe,DSP_DATAFRAME)
						if len(dataframe) > 20:
							if not os.path.isfile(dfname):											
								np.save(dfname,dataframe)
								print('Saved new Dataframe')
							else: 							
								fromfile = np.load(dfname)
								lst = np.append(fromfile, dataframe)
								np.save(dfname,lst)	
								print('Saved appending Dataframe')
							dataframe = []
	if len(dataframe): #if is not empty
		if os.path.isfile(dfname):
			fromfile = np.load(dfname)
			lst = np.append(fromfile, dataframe)
			np.save(dfname,lst)	
		else:
			np.save(dfname,dataframe)
		print('Saved last appending Dataframe')					
							
	return






def Create_DSP_BrightIII(source_path):
	# A Function that loads any .dsp Data from a given source Path and converts them into .npy Arrays for:
	# Training: 	If a Burst in the .dsp File according the the timestamp.ts File
	# Testing:		If not Burst is in it, so it can be used to Evaluate or Test the Data
	# Inputs:
	# 'source_path'		The Path Parent Path to the .dsp Files
	# Outputs:
	# NONE			It just stores files on the HD.

	idx = 0
	for path, subdir, files in os.walk(source_path):

		if not os.path.basename(os.path.normpath(path)) == 'Calibration':
			if files:
				#saveFolder(files,path,source_path)
				# Continue here If you Like to Test each Individual Sub-File
				for name in files:
					fname =  os.path.join(path, name)
					ext = os.path.splitext(fname)[1]
					if ext == '.dsp':
						getDSP_BrightIII(fname,path,source_path)
			
							
	return



def Fetch_DSP_Data(fname,DynamicSpectrum):
	# A Function to Read the DynamicSpectrum from an Array with a given mode_acq
	# Inputs:
	# 'fname' 				
	# 'DynamicSpectrum'
	# Outputs:

	if DynamicSpectrum['mode_acq'] == 1 or DynamicSpectrum['mode_acq'] == 2:
		#If DSP mode is 1 or 2 then the Spectrum Header is followed by data in the format of
		#SPECTRUM No1
		dt = np.dtype('int32')
		dt = dt.newbyteorder('>') # Change Endianity
		NSMPLS = np.frombuffer(DynamicSpectrum['dspARR'], dtype=dt, count=-1, offset=256) #all N-Samples as one long vector
		File_Size = os.path.getsize(fname)
		STS = 64+4096 #Size of one Time Sample in bytes, incl. Sample-Header and Sample-Datablock 
		TNS=(File_Size-256)/STS # Total Number of Samples
		TNS = np.floor(TNS) # If the fraction results in a floating point result
		TNS = TNS.astype(int) # Reshaping can only be done with integer numbers
		if not (NSMPLS.shape)/(TNS*(1024+16)) == 1: # This one is 
			NSMPLS = NSMPLS[:TNS*(1024+16)]
			msg = f"'{fname}': was incomplete, chopped it to reshape!"
			print(msg, end='\n')	
		data = np.transpose(NSMPLS.reshape(TNS,1024+16)) # reshape into a matrix: -> -> imagine the compelte data is stored in a single, very long vector and we know how many elements a sample has (they all have the same) and how many samples we have, so we can 'squish' (or fold) the vector into a matrix with exactly the column and row size of the number of samples and their elements!
		DSP_HEADER=data[:16,0:TNS] # separate the header from the matrix -> -> we can now seperate all sample header at once!
		DSP=data[17:,0:TNS] # separate the data from the matrix -> -> and can seperate all samples at once! 

		del DynamicSpectrum['dspARR'] # We do not need the HEX-Array no longer
		DynamicSpectrum['DSP_HEADER'], DynamicSpectrum['DSP'], DynamicSpectrum['TNS'] = DSP_HEADER, DSP, TNS
	else:
		print('DSP Mode 2 not yet supported')

	return DynamicSpectrum


def Load_DSP_File(fname):
	# A Function to read the comple .dsp file
	# Inputs:
	# 'fname'		the FULL path to the .dsp File
	# Outputs:
	# A Touple with the following Elements:
	# 'ARRAY'			the complete Array (to be processed later again)
	# 'fft_points' 	number of FFT points (256 or 1024)
	# 'mode_acq ' 	number and type of channels (1,2 or 3)
	# 'start_time' 	start time, as UNIX Timestamp
	# 'millisec '	duration of a spectrum in milliseconds (integer part)
	# 'FLow '		Frequency of the spectrum point 0 in Hz
	# 'FHigh'		Frequency of the spectrum point 1023 (or 255) in Hz
	# Read the file with 'READ BINARY' method
	with open(fname, "rb") as file:
		ARRAY = file.read()	

    #DSP FILE HEADER (one per FILE)
	dt = np.dtype('int32')
	dt = dt.newbyteorder('>') # Change Endianity 
	dsp_hdr = np.frombuffer(ARRAY, dtype=dt, count=20, offset=0)

	DynamicSpectrum = {	'dspARR' : ARRAY,
						'fft_points': dsp_hdr[6],
						'mode_acq' : dsp_hdr[11],
						'start_time' : dsp_hdr[13], 
						'millisec' : dsp_hdr[15], 
						'FLow' : dsp_hdr[16],
						'FHigh' : dsp_hdr[17]}

	return DynamicSpectrum


def str_arr_to_float_arr(str_arr):
    return [float(s.replace(',', '.')) for s in str_arr if s]

def float_matrix_to_list_of_numpy_vectors(float_matrix):
    list_np_vec = np.array(float_matrix)
    list_np_vec = list_np_vec.transpose()
    return list_np_vec

def update_block_PARAMS(block):
	dtfmt='%d.%m.%Y %H.%M.%S'
	bTimea = datetime.strptime(block['date'] + ' ' + block['time'], dtfmt)
	if not np.isnan(sum(block['parameters'][1])) and sum(block['parameters'][1]) > 45:
		bTimeb = bTimea + timedelta(seconds=sum(block['parameters'][1]))
	elif not np.isnan(sum(block['parameters'][1])) and sum(block['parameters'][1]) < 45:
		bTimeb = bTimea + timedelta(seconds=45)
	else:
		bTimeb = bTimea + timedelta(seconds=45)
	bTime = (bTimea.strftime(dtfmt),bTimeb.strftime(dtfmt))   
	block.update({'BurstTime': bTime})
	return block    


def TimeVector(DynamicSpectrum,whos):

	dtfmt='%d.%m.%Y %H.%M.%S'

	if whos == 'TEST':
		a = datetime.strptime(DynamicSpectrum['Total_Time'][0], dtfmt)
		b = datetime.strptime(DynamicSpectrum['Total_Time'][1], dtfmt)
	elif whos == 'TRAIN':
		a = datetime.strptime(DynamicSpectrum['BurstTime'][0], dtfmt)
		b = datetime.strptime(DynamicSpectrum['BurstTime'][1], dtfmt)		
	start = pd.Timestamp(a)
	end = pd.Timestamp(b)
	t = np.linspace(start.value, end.value, 6)
	t = pd.to_datetime(t)	
	TimeVector = []
	for i in t:
		i.to_pydatetime()	
		i = i.strftime('%H:%M:%S')
		TimeVector.append(i)


	return TimeVector


def timestampsIO(file_name):
	# A function to read the timestamps.ts file in blocks of measurement data.
    data = []
    try:
        with io.open(file_name, "r", encoding="utf8") as fp:

            header = fp.readline().strip().split("\t")
            data.append({"header": header[2:]})

            block = {}

            EOF = False
            while not EOF:
                for cnt in range(0, 6):
                    line = fp.readline()
                    # python reads an empty line at EOF
                    if not line:
                        EOF = True
                        break
                    if cnt == 0:
                        block["date"] = line.strip()
                    elif cnt == 1:
                        d = line.strip().split("\t")
                        block["time"] = d[0]
                        block["parameters"] = [str_arr_to_float_arr(d[1:])]
                    elif cnt == 2:
                        d = line.strip().split("\t")
                        block["type"] = d[0]
                        block["parameters"].append(str_arr_to_float_arr(d[1:]))
                    elif cnt in [3, 4, 5]:
                        d = line.strip().split("\t")
                        block["parameters"].append(str_arr_to_float_arr(d))
                        if cnt == 5:
                            # if you want to use numpy -> convert parameters here before append block
                            block["parameters"] = float_matrix_to_list_of_numpy_vectors(block.get("parameters", None))
                            update_block_PARAMS(block)
                            data.append(block)
                            block = {}
            return data                

    except (IOError, Exception) as ex:
        err_msg = f"Could not open file {file_name}.\nErrorMsg: {str(ex)}"
        print(err_msg)
        print("[NOK]")
        raise ex
    else:
        print("[OK]")



def plotDSP(DynamicSpectrum):	
	#A Function that Plots the DynamicSpectrum: Either complete Measurement or just Burst
	# Inputs:
	# 'DynamicSpectrum'
	# Outputs:
	# NONE


	if DynamicSpectrum['Label'] == 'TRAIN':
		fig, (ax0,ax1) = plt.subplots(ncols=2)

		img = cv2.resize(DynamicSpectrum['DSP'].astype('float64'), (1000,500), interpolation = cv2.INTER_CUBIC)		
		pc0 = ax0.pcolormesh(img,cmap='gist_ncar')
		ax0.set_yticklabels([8., 13., 18., 23., 28., 33.])
		ax0.set_ylabel('Frequency [kHz]')

		timevec = TimeVector(DynamicSpectrum,'TEST')
		ax0.set_xticklabels([])
		ax0.set_xticklabels(timevec)

		ax0.set_ylabel('Time [HH:MM:SS]')
		title = 'ORIGINAL' + '  ' + DynamicSpectrum['Total_Time'][0] + ' until '+ DynamicSpectrum['Total_Time'][1] + ' [UT]'
		ax0.set_title(title)


		img = cv2.resize(DynamicSpectrum['DSP_CUTTED'].astype('float64'), (1000,500), interpolation = cv2.INTER_CUBIC)		
		pc0 = ax1.pcolormesh(img,cmap='gist_ncar')
		ax1.set_yticklabels([8., 13., 18., 23., 28., 33.])
		ax1.set_ylabel('Frequency [kHz]')

		timevec = TimeVector(DynamicSpectrum,'TRAIN')
		ax0.set_xticklabels([])
		ax0.set_xticklabels(timevec)
		ax1.set_ylabel('Time [HH:MM:SS]')
		title = DynamicSpectrum['Label'] + '  ' + DynamicSpectrum['BurstTime'][0] + ' until '+ DynamicSpectrum['BurstTime'][1] + ' [UT]'
		ax1.set_title(title)
	else:
		title = DynamicSpectrum['Label'] + '  ' + DynamicSpectrum['Total_Time'][0] + ' until '+ DynamicSpectrum['Total_Time'][1] + ' [UT]'	
		fig, (ax0) = plt.subplots(ncols=1)

		img = cv2.resize(DynamicSpectrum['DSP'].astype('float64'), (1000,500), interpolation = cv2.INTER_CUBIC)		
		pc0 = ax0.pcolormesh(img,cmap='gist_ncar')
		ax0.set_yticklabels([8., 13., 18., 23., 28., 33.])
		ax0.set_ylabel('Frequency [kHz]')
		
		timevec = TimeVector(DynamicSpectrum,'TEST')
		ax0.set_xticklabels([])
		ax0.set_xticklabels(timevec)
		ax0.set_ylabel('Time [HH:MM:SS]')
		title = DynamicSpectrum['Label'] + '  ' + DynamicSpectrum['Total_Time'][0] + ' until '+ DynamicSpectrum['Total_Time'][1] + ' [UT]'
		ax0.set_title(title)

	plt.show()
	
	return

if os.path.isfile('/media/WDSSD/MachineLearning/UTR-2/savelog.txt'):
    open('/media/WDSSD/MachineLearning/UTR-2/savelog.txt', 'w').close()

source_path = '/media/WDSSD/MachineLearning/UTR-2/UTR-2 2002_forTesting'
#Create_DSP_NPY_data(source_path)
#Create_DSP_DATAFRAME(source_path)


#Create_DSP_BrightIII(source_path)

Create_DSP_IMAGE(source_path)


#Create_DATAFRAME(source_path)

