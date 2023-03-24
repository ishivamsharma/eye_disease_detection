#!/usr/bin/env python
# coding: utf-8

# #### We will try to make a **comparison** between building and training a **nerual network** and using **Transfer learning(pretrained model)**, in terms of **trainig time, ease of creation, accuracy and consistancy**

# In[3]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:


get_ipython().system('pip install tensorflow==V2.9.1')


# ## Importing libraries

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.metrics import plot_confusion_matrix
warnings.filterwarnings('ignore')
get_ipython().system('pip install visualkeras')
import visualkeras


# In[6]:


# import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# for gpu in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu, True)


# In[7]:


# generating dataset from directory

# Generating train dataset
data = tf.keras.utils.image_dataset_from_directory(directory = '/content/drive/MyDrive/Colab Notebooks/dataset',
                                                   color_mode = 'rgb',
                                                   batch_size = 64,
                                                   image_size = (224,224),
                                                   shuffle=True,
                                                   seed = 2022)


# ## Displaying data distribution

# In[8]:


labels = np.concatenate([y for x,y in data], axis=0)


# In[9]:


values = pd.value_counts(labels)
values = values.sort_index()


# In[10]:


values


# ### Checking labels

# In[11]:


# getting class names
class_names = data.class_names
for idx, name in enumerate(class_names):
  print(f"{idx} = {name}", end=", ")


# ### The data is well distributed among the classes and is balanced

# In[12]:


plt.figure(figsize=(12,8))
plt.pie(values,autopct='%1.1f%%', explode = [0.02,0.02,0.02, 0.02], textprops = {"fontsize":15})
plt.legend(labels=data.class_names)
plt.show()


# ## Getting a data generator to explore the data

# In[13]:


data_iterator = data.as_numpy_iterator()


# In[14]:


batch = data_iterator.next()


# ### Each batch contains 64 images, each image is 224x224

# In[15]:


batch[0].shape


# ## Displaying some images

# In[16]:


plt.figure(figsize=(10, 10))
for images, labels in data.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# # Preprocessing images

# ## Standardize the data

# In[17]:


data = data.map(lambda x, y: (x/255, y))


# In[18]:


sample = data.as_numpy_iterator().next()


# In[19]:


print(sample[0].min())
print(sample[0].max())


# ## Spliting the data

# In[20]:


print("Total number of batchs = ",len(data))


# In[21]:


train_size = int(0.7 * len(data)) +1
val_size = int(0.2 * len(data))
test_size = int(0.1 * len(data))


# In[22]:


train = data.take(train_size)
remaining = data.skip(train_size)
val = remaining.take(val_size)
test = remaining.skip(val_size)


# In[23]:


print(f"# train batchs = {len(train)}, # validate batchs = {len(val)}, # test batch = {len(test)}")
len(train) + len(val) + len(test)


# ## Preparing test set

# In[24]:


test_iter = test.as_numpy_iterator()


# In[25]:


test_set = {"images":np.empty((0,224,224,3)), "labels":np.empty(0)}
while True:
    try:
        batch = test_iter.next()
        test_set['images'] = np.concatenate((test_set['images'], batch[0]))
        test_set['labels'] = np.concatenate((test_set['labels'], batch[1]))
    except:
        break


# In[26]:


y_true = test_set['labels']


# # CNN from scratch

# In[27]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization
from keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Adamax


# In[28]:


# Displaying history loss/accuracy
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot_his(history):
    plt.figure(figsize=(15,12))
    metrics = ['accuracy', 'loss']
    for i, metric in enumerate(metrics):
        plt.subplot(220+1+i)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
    plt.show()


# In[29]:


def create_baselineCNN():
    model = Sequential([
        Conv2D(filters = 64, kernel_size=3, activation = 'relu',padding='same', input_shape=(224,224,3)),
        Conv2D(filters = 64, kernel_size=3, activation = 'relu',padding='same'),
        BatchNormalization(),
        MaxPool2D(2),
        Dropout(0.3),

        Conv2D(filters = 128, kernel_size=3,padding='same', activation = 'relu',),
        Conv2D(filters = 128, kernel_size=3,padding='same', activation = 'relu',),
        BatchNormalization(),
        MaxPool2D(2),
        Dropout(0.3),
        
        Conv2D(filters = 128, kernel_size=3,padding='same', activation = 'relu',),
        Conv2D(filters = 128, kernel_size=3,padding='same', activation = 'relu',),
        BatchNormalization(),
        Conv2D(filters = 128, kernel_size=3,padding='same', activation = 'relu',),
        BatchNormalization(),
        MaxPool2D(2),
        Dropout(0.3),
        
        Flatten(),
        Dense(64, activation = 'relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dense(128, activation = 'relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dense(4, activation='softmax')
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    return model


# In[30]:


model = create_baselineCNN()


# In[31]:


model.summary()

visualkeras.layered_view(model,legend=True)


# ## Model training

# In[32]:


from keras import callbacks 
early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy", 
        patience=20,
        verbose=1,
        mode="max",
        restore_best_weights=True, 
     )

history = model.fit(
    train,
    validation_data=val,
    epochs = 60,
    callbacks=[early_stop],
)


# In[33]:


plot_his(history)


# ## Evaluating the model on the test set

# In[34]:


y_pred = np.argmax(model.predict(test_set['images']), 1)


# In[35]:


print(classification_report(y_true, y_pred, target_names = class_names))


# In[36]:


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
plt.xticks(np.arange(4)+.5, class_names, rotation=90)
plt.yticks(np.arange(4)+.5, class_names, rotation=0)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")


# # Insights
# #### The model acheived 90% accuracy
# #### The training has run for 60 epochs and did not stop by early stopping
# #### Although it achieved a good accuracy but validation loss wasn't consistant at all and has a lot of fluctuations

# -----------
# # Transfer Learning (Pretrained Model)
# #### Here we will use a pretraind model and finetune it to fit our data
# #### We are using EfficientNet pretrained model

# In[37]:


def make_model():
    effnet = EfficientNetB3(include_top=False, weights="imagenet",input_shape=(224,224,3), pooling='max') 
    effnet.trainable=False
    
    for layer in effnet.layers[83:]:
      layer.trainable=True
    
    x = effnet.output
    x = BatchNormalization()(x)
    x = Dense(1024, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
                    bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)
    x = Dropout(rate=.45, seed=2022)(x)        
    output=Dense(4, activation='softmax')(x)
    
    model= tf.keras.Model(inputs=effnet.input, outputs=output)
    model.compile(optimizer = 'adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model


# In[38]:


model = make_model()


# ## Train the Model

# In[39]:


from keras import callbacks 

save_best_only = True #to save best model file
from keras.callbacks import ModelCheckpoint , EarlyStopping

mc = ModelCheckpoint(filepath='/content/drive/MyDrive/Project',                     
                     verbose=1,
                     save_best_only = True) 


early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy", 
        patience=10,
        verbose=1,
        mode="max",
        restore_best_weights=True, 
     )

cb = [mc , early_stop]

history = model.fit(
    train,
    validation_data=val,
    epochs = 50,
    callbacks=[mc ,early_stop]
)


# In[40]:


plot_his(history)


# ## Evaluating the model on test set

# In[41]:


y_pred = np.argmax(model.predict(test_set['images']), 1)


# In[42]:


print(classification_report(y_true, y_pred, target_names = class_names))


# In[43]:


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
plt.xticks(np.arange(4)+.5, class_names, rotation=90)
plt.yticks(np.arange(4)+.5, class_names, rotation=0)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")


# # Insights
# #### The model acheived 93% accuracy
# #### The training has run for 47 epochs and stopped by early stopping
# #### The validation loss curve looks very smooth and has no fluctuations

# In[44]:


import tensorflow as tf

from keras.models import load_model

model.save('TL.hdf5')


# In[45]:


tf.keras.models.save_model(model, "TL.hdf5")


# In[46]:


get_ipython().system('pip install streamlit')


# In[47]:


get_ipython().system('killall ngrok')


# In[48]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st\nimport tensorflow as tf\n\nst.set_option(\'deprecation.showfileUploaderEncoding\',False)\n@st.cache(allow_output_mutation=True)\ndef load_model():\n    model = tf.keras.models.load_model(\'/content/TL.hdf5\')\n    return model\nmodel = load_model()\n\nst.write("""\n         # Eye Disease Detection \n         """\n        )\n\nfile = st.file_uploader("Please upload the image of eye",type=["jpg","png"])\n#import cv2\nfrom PIL import Image,ImageOps\nimport numpy as np\n\ndef import_and_predict(img,model):\n    size = (224,224)\n    image = ImageOps.fit(img,size,Image.ANTIALIAS)\n    img = np.asarray(image)\n    img_reshape = img[np.newaxis,...]\n    prediction = model.predict(img_reshape)\n    \n    return prediction\n\nif file is None:\n    st.text("Please upload an image")\nelse:\n    image = Image.open(file)\n    st.image(image,use_column_width=True)\n    predictions = import_and_predict(image,model)\n    class_names= [\'Cataract\', \'Diabetic_retinopathy\', \'Glaucoma\', \'Normal\']\n    string = "This is image is most likely is : "+class_names[np.argmax(predictions)]\n    st.success(string)')


# In[49]:


get_ipython().system('ngrok authtoken 2Mv4GPuvYY8Fcbm9CiNTldOd6GI_3nchiqZqvTQ1VKSnKssDD')


# In[50]:


get_ipython().system('pip install pyngrok')


# In[ ]:


get_ipython().system('ngrok http 80')


# In[ ]:


get_ipython().system('nohup streamlit run app.py &')


# In[ ]:


import streamlit
get_ipython().system('streamlit run --server.port 80 app.py >/dev/null')


# In[ ]:


# from pyngrok import ngrok 
# public_url = ngrok.connect(port='8501')
# public_url


# In[ ]:


# !cat /content/nohup.out


# In[ ]:


get_ipython().system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')


# In[ ]:


get_ipython().system('unzip ngrok-stable-linux-amd64.zip')


# In[ ]:


get_ipython().system_raw('./ngrok http 8501 &')


# In[ ]:


get_ipython().system('curl -s http://localhost:4040/api/tunnels | python3 -c     \'import sys, json; print("Execute the next cell and the go to the following URL: " +json.load(sys.stdin)["tunnels"][0]["public_url"])\'')


# In[ ]:


get_ipython().system('streamlit run /content/app.py')


# # Conclusion
# 
# #### It's very intersting how transfer learning can improve and satisfy our goals with higher accuracy, consistent and faster
# #### With pretraind model, we achieved 93% with 7% improvment from our own Neural network 90%

# In[ ]:




