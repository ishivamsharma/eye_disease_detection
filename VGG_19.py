#!/usr/bin/env python
# coding: utf-8

# In[2]:



from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator , load_img , img_to_array


# In[ ]:





# In[ ]:





# In[5]:


# re-size all the images to this
IMAGE_SIZE = [224, 224 ]


# In[6]:


train_path = 'E:\DBDA PRJ\Training'
valid_path = 'E:\DBDA PRJ\Testing'


# In[7]:


# add preprocessing layer to the front of VGG
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[5]:


# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False


# In[13]:


# useful for getting number of classes
folders = glob('E:\DBDA PRJ\Training\*')


# In[14]:


# our layers - you can add more if you want
x = Flatten()(vgg.output)


# In[15]:


# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)


# In[16]:


# create a model object
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()
#model = np.squeeze(model)


# In[10]:


# view the structure of the model


# In[11]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[36]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('E:\DBDA PRJ\Training',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('E:\DBDA PRJ\Testing',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[13]:


# for image_batch, label_batch in train_generator:
#       break
# image_batch.shape, label_batch.shape


# In[ ]:





# In[ ]:





# In[14]:


training_set.class_indices  #to print classes of dataset


# In[15]:


save_best_only = True #to save best model file
from keras.callbacks import ModelCheckpoint , EarlyStopping

mc = ModelCheckpoint(filepath='E/vgg19/',                     
                     verbose=1,
                     save_best_only = True) 

ec = EarlyStopping(
                  min_delta=0.01,
                  patience=5,
                  verbose=1)
                   
    
cb = [mc , ec]


# In[16]:


# fit the model
r = model.fit_generator(
              training_set,
              validation_data=test_set,
              epochs=100,
              callbacks=cb,
              steps_per_epoch=len(training_set),
              validation_steps=len(test_set)
             )


# In[ ]:





# In[25]:


# loss
plt.plot(r.history['loss'], label='train loss', color='red')
plt.plot(r.history['val_loss'], label='val loss',color='blue')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')


# In[28]:


# accuracies
plt.plot(r.history['accuracy'], label='train acc',color='red')
plt.plot(r.history['val_accuracy'], label='val acc',color='blue')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[4]:


import tensorflow as tf

from keras.models import load_model

model.save('vgg19.h5')


# In[27]:





# In[ ]:





# In[5]:



path = 'Solid_white.svg.webp'
img = load_img(path,target_size=(224,224))
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
elif pred == 5:
    print("the image belong to Uveitis disease")
else:
    print("the eye is not detected")
                        
                        
plt.imshow(input_arr[0])
plt.title("input image")
plt.axis=False
plt.show()




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




