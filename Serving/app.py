import base64
import csv
import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import pickle
from fastapi.responses import JSONResponse
import pandas as pd
import sklearn
import uvicorn
from pydantic import BaseModel
import subprocess

# Define a Pydantic model to parse the incoming JSON data
class FeedbackData(BaseModel):
    image: str  # Base64 encoded image
    prediction: str
    target: str

app = FastAPI()


#load the trained model, pca
with open("/Artifacts/modele.pkl","rb") as f:
    model = pickle.load(f)
with open("/Artifacts/pca.pkl","rb") as f:
    pca = pickle.load(f)



#MINIMAL APP - GET REQUEST
@app.get("/", tags=['ROOT'])
async def root() -> dict: 
    return{"message": "Bienvenue sur l'API de détection d'anomalies de céramiques"}



@app.post("/predict")
async def predict(file: UploadFile):

    # Prétraiter l'image avant de la passer au modele 
    image = Image.open(file.file)
    image_np = np.array(image).flatten().reshape(1,-1)
    image_np = image_np/255
    pca_image = pca.transform(image_np)

    # Renvoyer la prediction 
    prediction = model.predict(pca_image)
    label = "anomalie" if prediction[0] == 1 else "good"
    
    return JSONResponse(content={"prediction": label})




@app.post("/feedback")
async def feedback(feedback_data: FeedbackData):

    # prétraitement de l'image
    image_data = base64.b64decode(feedback_data.image)
    image = Image.open(io.BytesIO(image_data))
    image_np = np.array(image).flatten().reshape(1, -1)
    scaled_image = image_np/255
    pca_image = pca.transform(scaled_image)

    # Noms de colonnes
    num_columns = pca.n_components_
    column_names = ['PCA_' + str(i) for i in range(num_columns)] + ['prediction', 'target']

    # Process other feedback data
    prediction = feedback_data.prediction
    actual_target = feedback_data.target

    # Data to be written to the CSV file
    pca_image = pca_image.flatten().tolist()
    data_row = pca_image + [ 1 if prediction =='anomalie' else 0, 1 if actual_target=='Oui' else 0]

    #récupere le chemin du fichier prod_data (si existe).
    script_dir = os.path.dirname(__file__)
    prod_data_path = os.path.join(script_dir, '../Data/prod_data.csv')
    file_exists = os.path.isfile(prod_data_path)
    with open(prod_data_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(column_names)
        writer.writerow(data_row)



    #  Reentrainement du modele, en fonction du trigger k.
    pathProdData = "/Data/prod_data.csv"
    K = 10
    if shouldUpdateModel(pathProdData,K):
        updateModel()


    # Retourner une reponse
    response_data = {
        "message": "Feedback received successfully",
        "prediction": feedback_data.prediction,
        "actual_target": feedback_data.target
    }


    return JSONResponse(status_code=200, content=response_data)


def shouldUpdateModel(path, k ):
    """
    Verifie si le nombre de lignes dans csv si c'est un multiple de k
    """
    with open(path, 'r') as file:
        rowCount = sum(1 for row in file)
    return  (rowCount - 1) % k == 0

def updateModel():
    prod_data_path = '/Scripts/modele.py'
    subprocess.run(['python', prod_data_path])
