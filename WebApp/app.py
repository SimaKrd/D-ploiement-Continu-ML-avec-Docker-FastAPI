import base64
from io import BytesIO
import streamlit as st
from PIL import Image
import requests

# Convert BytesIO to a base64 encoded string
def convert_image_to_base64(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format=image.format)
    encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    return encoded_image

# Titre de l'application
st.title("Ceramic anomaly detection")

# Sélectionner une image à partir de l'ordinateur
uploaded_file = st.file_uploader("Importer une image", type=["jpg", "png", "jpeg"])

#Initialisation Variable de Session
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = False


if uploaded_file is not None:

    image = Image.open(uploaded_file)
    DImage = convert_image_to_base64(image)
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}


    st.image(image, caption='Image téléchargée.', use_column_width=True)

    # Bouton pour extraire le texte de l'image
    if st.button('Prédire'):
        # envoie l'image a FastAPI
        response = requests.post("http://serving_api:8080/predict", files={"file": img_byte_arr})
        result = response.json()
        st.write("Réponse de l'API FastAPI :")
        #st.write(result)

        # Display the prediction result
        label = result["prediction"]
        text_output = st.empty()
        text_output.text(label)

        st.session_state['pred_result'] = label
        st.session_state['prediction'] = True





# Gestion du feedback 

if st.session_state['prediction'] : 


    st.write("La prédiction était-elle correcte ?")
    feedback_options = ['Oui', 'Non']
    user_feedback = st.radio("Choisissez une option   :" , feedback_options)
    
    if st.button('Soumettre le feedback'):
        feedback_data= {
            'image' : DImage,
            'prediction' : st.session_state['pred_result'],
            'target' : user_feedback
        }

        response = requests.post("http://serving_api:8080/feedback", json=feedback_data)

        if response.ok :
            if user_feedback == 'Oui' :
                st.success("Super ! Merci pour le retour.")
            else:
                st.error("Oh mince ! Merci pour le retour.")

        else:
            st.error('Erreur lors de la transmission du feedback.')
        



