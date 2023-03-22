#!/usr/bin/env python
# coding: utf-8

# In[6]:


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

#import cv2
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator , load_img , img_to_array
#from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
#pip install "tensorflow>=1.7.0"
import os
import tensorflow_hub as hub
import image as val_image_batch
from keras.models import load_model


# In[5]:





# In[7]:



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


# In[8]:


tf.__version__


# In[9]:


import pandas as pd

# Increase precision of presented data for better side-by-side comparison
pd.set_option("display.precision", 8)


# In[11]:


data_root ='E:\DBDA PRJ'


# In[12]:


IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = str(data_root)

datagen_kwargs = dict(rescale=1./255, validation_split=.20)
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


# In[13]:



for image_batch, label_batch in train_generator:
      break
image_batch.shape, label_batch.shape


# In[14]:


print (train_generator.class_indices)

labels = '\n'.join(sorted(train_generator.class_indices.keys()))

with open('labels.txt', 'w') as f:
      f.write(labels)


# In[9]:



#!eye_disease_labels.txt


# In[15]:


IMAGE_SIZE = 224*224


# In[ ]:





# In[16]:


model = tf.keras.Sequential([
  hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", 
                 output_shape=[1280],
                 trainable=False),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])
model.build([None, 224, 224, 3])

model.summary()
     


# In[17]:


len(model.trainable_variables)


# In[18]:


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


# In[19]:


model.compile(
  optimizer=optimizer,
  loss='categorical_crossentropy',
  metrics=['acc'])


# In[20]:


from keras.callbacks import ModelCheckpoint , EarlyStopping

mc = ModelCheckpoint(filepath='E:\model1',
                     monitor="val_acc",                     
                     verbose=1,
                     save_best_only = True) 

ec = EarlyStopping(monitor= 'val_acc',
                  min_delta=0.01,
                  patience=5,
                  verbose=1)
                   
    
cb = [mc , ec]


# In[21]:


loss, acc = model.evaluate(train_generator)


# In[22]:


steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)
val_steps_per_epoch = np.ceil(valid_generator.samples/valid_generator.batch_size)


# In[18]:




hist = model.fit(
    train_generator, 
    epochs=100,
    verbose=1,
    callbacks=cb,
    steps_per_epoch=steps_per_epoch ,
    validation_data=valid_generator,
    validation_steps=val_steps_per_epoch)
     


# In[86]:


h = hist.history
h.keys()


# In[96]:


final_loss, final_accuracy = model.evaluate(valid_generator, steps = val_steps_per_epoch)
print("Final loss: {:.2f}".format(final_loss))
print("Final accuracy: {:.2f}%".format(final_accuracy * 100))


# In[93]:


plt.plot(h['loss'] , 'g--' , color ="blue")
plt.plot(h['acc'] ,  'g--' , color ="red")
plt.title("LOSS VS ACCURACY")
plt.show()


# In[94]:


# plt.figure()
# plt.ylabel("Loss (training and validation)")
# plt.xlabel("Training Steps")   
       
# plt.ylim([0,50])
# plt.plot(hist[""])
# plt.plot(hist[""]) 

# plt.figure()
# plt.ylabel("Accuracy (training and validation)")
# plt.xlabel("Training Steps")
# plt.ylim([0,1])
# plt.plot(hist["final_loss"])
# plt.plot(hist["final_accuracy"])
# plt.show()
     


# In[27]:


val_image_batch, val_label_batch = next(iter(valid_generator))
true_label_ids = np.argmax(val_label_batch, axis=-1)
print("Validation batch shape:", val_image_batch.shape)


# In[28]:


dataset_labels = sorted(train_generator.class_indices.items(), key=lambda pair:pair[1])
dataset_labels = np.array([key.title() for key, value in dataset_labels])
print(dataset_labels)


# In[29]:


tf_model_predictions = model.predict(val_image_batch)
print("Prediction results shape:", tf_model_predictions.shape)
     


# In[30]:


predicted_ids = np.argmax(tf_model_predictions, axis=-1)
predicted_labels = dataset_labels[predicted_ids]
print(predicted_labels)


# In[31]:


plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range((len(predicted_labels)-2)):
  plt.subplot(6,5,n+1)
  plt.imshow(val_image_batch[n])
  color = "green" if predicted_ids[n] == true_label_ids[n] else "red"
  plt.title(predicted_labels[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
     


# In[23]:


from keras.models import load_model

model = load_model('E:\model1')


# In[24]:


model.save_weights("E:\model1/final_weights.h5")


# In[54]:


h = hist.history
h.keys()


# In[55]:


from keras.models import Model 
from keras.applications.mobilenet_v2 import   preprocess_input
import keras


# In[64]:


kwargs={'training': 'True'}
import numpy as np


# In[66]:


# predictions = model.predict(data)
# print('Shape: {}'.format(predictions.shape))

# from keras.applications.mobilenet_v2 import decode_predictions

# for name, desc, score in decode_predictions(predictions)[0]:
#     print('- {} ({:.2f}%%)'.format(desc, 100 * score))


# In[84]:


path = '5621050615_85cc77061a_o.jpg'
img = load_img(path,target_size=(224,224))
i = img_to_array(img)

i = preprocess_input(i) #pre process the input image 

input_arr = np.array([i])

input_arr.shape


pred = np.argmax(model.predict(input_arr))
#pred = np.argmax(model.predict_on_batch(input_arr))

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





# In[ ]:





# In[ ]:




