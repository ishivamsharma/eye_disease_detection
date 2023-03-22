#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from shutil import copy2
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K
#from keras.optimizers import Adam,RMSprop
import tensorflow as tf

#import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
import matplotlib as plt
import numpy as np


import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator , load_img , img_to_array 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout , GlobalAveragePooling2D
#pip install "tensorflow>=1.7.0"
import os
import tensorflow_hub as hub
#import image as val_image_batch


# In[3]:



from __future__ import absolute_import, division, print_function, unicode_literals

try:
  # The %tensorflow_version magic only works in colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


# In[4]:



import pandas as pd
# Increase precision of presented data for better side-by-side comparison
pd.set_option("display.precision", 8)


# In[4]:


#data path
data_root ='E:\DBDA PRJ'


# In[5]:


#use image datagen to detect data
IMAGE_SHAPE = (256, 256 )
TRAINING_DATA_DIR = str(data_root)

datagen_kwargs = dict(rescale=1./255, validation_split=.20 )
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    TRAINING_DATA_DIR, 
    subset="validation", 
    shuffle=True,
    target_size=IMAGE_SHAPE
)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DATA_DIR, 
    subset="training", 
    shuffle=True,
    target_size=IMAGE_SHAPE)


# In[6]:


for image_batch, label_batch in train_generator:
      break

        image_batch.shape, label_batch.shape


# In[7]:


#save all the label of classes
print (train_generator.class_indices)

labels = '\n'.join(sorted(train_generator.class_indices.keys()))

with open('labels.txt', 'w') as f:
      f.write(labels)


# In[8]:


IMAGE_SIZE = 224*224


# In[38]:


from keras.models import Model
from keras.applications.inception_v3 import InceptionV3 , preprocess_input
import keras


# In[10]:


base_model = InceptionV3(weights='imagenet',input_shape=(256,256,3),  include_top=False,classes=6 )


# In[11]:


#stop training of top layer
for layer in base_model.layers:
    layer.trainable = True


# In[12]:


x = Flatten()(base_model.output)
x = Dense(units=6 , activation= 'softmax')(x)


# In[13]:


#create model
model = Model(base_model.input , x)


# In[15]:


model.compile( 
  optimizer= 'Adam',
  loss='categorical_crossentropy',
  metrics=['acc'])


# In[16]:


#summary
model.summary()


# In[60]:


train_datagen = ImageDataGenerator(
                                   rotation_range=0.1,
                                   width_shift_range=0.3,
                                   horizontal_flip=True,
                                   preprocessing_function= preprocess_input , 
                                   zoom_range=0.1,
                                   shear_range=0.4,
                                    )


# In[61]:


train_data = train_datagen.flow_from_directory(directory= 'E:\DBDA PRJ' , 
                                               target_size= (256,256),
                                               batch_size=34, #change it to 32 to remove error
                                               subset='training',
                                               shuffle=True)


# In[19]:


train_data.class_indices  #check the classs image


# In[20]:


t_img , label = train_data.next()


# In[21]:


maxValue = np.amax(t_img)
print(maxValue)
minValue = np.amin(t_img)
print(minValue)


# In[22]:


t_img.shape  #to check batch size


# In[23]:


#input = image array
#output = plot image

def plotImages(img_arr , label):
    for idex , img in enumerate(img_arr):
        plt.figure(figsize=(5,5))
        plt.imshow(img, vmin=1, vmax=255)   #plt.imshow(out, vmin=0, vmax=255)
        plt.title(img.shape)
        plt.axis = False
        plt.show()
        

plotImages(t_img , label)
        


# In[24]:


#model check point
#filepath ' path//' to store best model
save_best_only = True to save best model file
from keras.callbacks import ModelCheckpoint , EarlyStopping



mc = ModelCheckpoint(filepath='E:\model',
                     monitor="acc",                     
                     verbose=1,
                     save_best_only = True) 


# In[25]:


#early stoping to avoid overfitting



ec = EarlyStopping(monitor= 'val_acc',
                  min_delta=0.01,
                  patience=5,
                  verbose=1)


# In[26]:


#for call back 
cb = [mc , ec]


# In[27]:


#loss, acc = model.evaluate(train_data)


# In[28]:




his = model.fit(train_data, 
                steps_per_epoch=20,
                epochs=100,
                callbacks=cb
               
                         )
                          


# In[62]:


steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)
val_steps_per_epoch = np.ceil(valid_generator.samples/valid_generator.batch_size)


# In[ ]:





# In[1]:


from keras.models import load_model
import os


# In[ ]:





# In[7]:


h = his.history
h.keys()


# In[5]:


#PLOT A LOSS AND ACCURACY GRAPH
plt.plot(h['loss'] , 'g--' , color ="blue")
plt.plot(h['acc'] ,  'g--' , color ="red")
plt.title("LOSS VS ACCURACY")
plt.show()


# In[4]:


#validate code 
#from keras.applications.I import preprocess_input    
path = '5621050615_85cc77061a_o.jpg'
img = load_img(path,target_size=(256,256))
i = img_to_array(img)

i = preprocess_input(i) #pre process the input image 

input_arr = np.array([i])

input_arr.shape


pred = np.argmax(model.predict(input_arr))


if pred == 0:
   print("the image belong to Bulging_Eyes disease ")
elif pred == 1:
   print("the image belong to Cataract disease ")
elif pred == 2:
   print("the image belong to Crossed_Eyes disease")
elif pred == 3:
   print("the image belong to Glaucoma disease")
elif pred == 4:
   print("yor are NORMAL")
else:
   print("the image belong to Uveitis disease")
                       
                       
plt.imshow(input_arr[0])
plt.title("input image")
plt.axis=False
plt.show()
                   
   


# In[ ]:


# {'Bulging_Eyes': 0,
#  'Cataracts': 1,
#  'Crossed_Eyes': 2,
#  'Glaucoma': 3,
#  'NORMAL': 4,
#  'Uveitis': 5}


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




