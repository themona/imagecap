#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import re
import nltk
from nltk.corpus import stopwords
import string
import json
from time import time
import pickle
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.layers import add


# In[2]:


model = load_model("./model_weights/model_19.h5")
#model.predict_classes()

# In[3]:


model_temp = ResNet50(weights="imagenet", input_shape=(224,224,3))
model.summary()


# In[4]:


model_resnet = Model(model_temp.input,model_temp.layers[-2].output)
#model_resnet.predict_classes()

# In[17]:


def preprocess_image(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    # Normalisation
    img = preprocess_input(img)
    return img


# In[18]:


def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1,feature_vector.shape[1],)
    return feature_vector


# In[26]:





# In[27]:





# In[21]:


def predict_caption(photo):
    MAX_LEN=35
    in_text = "startseq"
    for i in range(MAX_LEN):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=MAX_LEN,padding='post')
        
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() #WOrd with max prob always - Greedy Sampling
        word = idx_to_word[ypred]
        in_text += (' ' + word)
        
        if word == "endseq":
            break
    
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption


# In[22]:


with open("saved/word_to_idx.pkl", "rb") as w2i:
    word_to_idx = pickle.load(w2i)


# In[23]:


with open("saved/idx_to_word.pkl", "rb") as i2w:
    idx_to_word = pickle.load(i2w)


# In[28]:

def caption_this_image(image):
    enc=encode_image(image)
    caption=predict_caption(enc)
    return caption


# In[ ]:




