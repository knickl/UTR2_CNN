#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np       
from tqdm import tqdm
import matplotlib.pyplot as plt


# In[2]:


cm0 = np.array([[4.0,6.0],[0.0,10.0]])
#cm1 = np.array([[5.0,2.0],[3.0,7.0]])
#cm2 = np.array([[13.0,2.0],[1.0,11.0]])
#cm3 = np.array([[13.0,2.0],[1.0,11.0]])
#cm = np.array([cm0,cm1, cm2, cm3])
cm = cm0


# In[3]:


def plotConfusionMatrix(cm):
    plt.rcParams["figure.figsize"] = (10,10)

    plt.subplot(1, 2, 1)
    classes = ['Burst','No Burst']
    cmap=plt.cm.Blues
    fmt = '.2f'
    thresh = cm.max() / 2.
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix: Object Detection')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()

    
    fname = '/home/stefan/Dokumente/EPSC_2019/Images/Final/confunsionmatrix_ Object Detection.png'

    plt.savefig(fname, dpi=400, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)

    
    plt.show()


# In[4]:


def calcConfusionMatrix(cm):

	tp = cm[0,0]
	tn = cm[1,1]
	fp = cm[1,0]
	fn = cm[0,1]

	#Accuracy
	Accuracy = (tn+tp)/(tp+tn+fp+fn) 
	print('Accuracy: %0.2f'%Accuracy)

	#Precision 
	ppv = tp/(tp+fp)
	print('PPV or Precision: %0.2f'%ppv)

	#Recall 
	Recall = tp/(tp+fn) 
	print('Sensitivity or Recall: %0.2f'%Recall)

	#Specificity 
	Specificity = tn/(tn+fp)
	print('Specificity or Selectivity : %0.2f'%Specificity)


	#F1 Score
	f1 = (2*ppv*Recall)/(ppv + Recall)
	print('F1 Score: %0.2f'%f1)


	#PPV
	npv = tn/(tn+fn)
	print('NPV: %0.2f'%npv)

calcConfusionMatrix(cm0)


# In[5]:


plotConfusionMatrix(cm)


# In[ ]:





# In[ ]:




