import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 
import os
from io import BytesIO,StringIO
import sys
import pandas as pd


st.header('Image Classification Model')
model = load_model('C:/Users/Rutuja/Desktop/pomo/final pomo/Image_classify.keras')
data_cat=['Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Cercospora','healthy']

img_height = 180
img_width = 180

image =st.text_input('Enter Image name','anthracnose.jpg')



image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write('the pomegranate in image is ' + data_cat[np.argmax(score)])
st.write('With accuracy of ' + str(np.max(score)*100))
    



