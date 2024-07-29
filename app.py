import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 
import google.generativeai as genai

genai.configure(api_key="AIzaSyCVC3Qp0TAXWtnyD95olAe0CL2JbkR_wHs")
generation_config = {
    "temperature": 0.9,
    "max_output_tokens": 2048,
    "top_k": 1,
    "top_p": 1,
}




st.header('Image Classification Model')
model = load_model('Image_classify.keras')
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
detect = data_cat[np.argmax(score)]
st.write('the pomegranate in image is ' + detect)
st.write('With accuracy of ' + str(np.max(score)*100))
    



match detect:
    case "Bacterial_Blight":
        
        st.write('Bacterial Blight ')
        st.write('Caused by: Xanthomonas axonopodis pv. punicae')
        st.write('Symptoms: Water-soaked spots on leaves, fruit, and stems that eventually turn dark brown or black.')
        st.write('Control Measures:\n- Copper-based Fungicides: These are effective against bacterial diseases.\n    - Examples: Copper hydroxide, copper oxychloride.\n    - Usage: Apply as a foliar spray at 10-15 day intervals, starting at the onset of symptoms. Follow label instructions for concentration.\n\n- Streptomycin:An antibiotic used for controlling bacterial infections.\n    - Usage: Apply as a foliar spray according to the label recommendations. Avoid excessive use to prevent resistance.\n\n- Cultural Practices:\n    -Prune and destroy affected plant parts to reduce inoculum.\n    -Avoid overhead irrigation to reduce leaf wetness.' )     

    case "Anthracnose":
        st.write('Anthracnose')   
        st.write('Caused by:Colletotrichum gloeosporioides')   
        st.write('Symptoms: Dark, sunken lesions on fruit, leaves, and stems, often with pink spore masses in moist conditions.')   
        st.write('Control Measures:\n- Fungicides: Broad-spectrum fungicides are effective against anthracnose.\n  - *Examples:* Chlorothalonil, mancozeb, carbendazim.\n- *Usage:* Apply as a foliar spray during wet conditions and repeat every 7-10 days if needed.- Cultural Practices:\n  - Remove and destroy affected plant parts.\n  - Ensure proper irrigation practices to avoid excessive moisture on plant surfaces.')

    case "Alternaria":
        st.write('Caused by: Alternaria alternate') 
        st.write('Symptoms: Dark brown to black spots on leaves, often with concentric rings and yellow halos.') 
        st.write('Control Measures:\n- Fungicides:Systemic and contact fungicides can control Alternaria.\n  - Examples: Difenoconazole, iprodione, pyraclostrobin.\n - Usage: Apply as a foliar spray at the onset of symptoms and repeat every 10-14 days as necessary.\n- Cultural Practices:\n  - Remove and destroy infected plant debris.\n  - Practice crop rotation and avoid planting in areas with previous infections.')

    case "Cercospora":
        st.write('Cercospora')
        st.write('Caused by: Cercospora punicae') 
        st.write('Symptoms: Small, circular to irregular brown spots on leaves, which can cause premature defoliation.') 
        st.write('Control Measures:*\n- Fungicides: Systemic fungicides are effective against Cercospora.\n  - Examples: Propiconazole, tebuconazole, azoxystrobin.\n  - Usage: Apply as a foliar spray at the first sign of disease and repeat every 10-14 days as needed.\n- Cultural Practices:\n - Remove and destroy infected leaves.\n  - Maintain good air circulation around plants by proper spacing and pruning.')

    case _:
        st.write('its healthy')     





model = genai.GenerativeModel("gemini-pro",generation_config=generation_config)
response = model.generate_content(["Give me summary forecast of solapur , from timeanddate website"])
st.write('' + response.text )
