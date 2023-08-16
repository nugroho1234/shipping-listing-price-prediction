from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from ship_price_variables import ShipPricePred

from tensorflow.keras.models import save_model, load_model
import pandas as pd
import pickle
import tensorflow as tf
import helper

ShipPricePredApp = FastAPI()
#load reconstructed inference model
reconstructed_model = load_model("model/inference_model.keras")

#load tfidf_vectorizer, transform the description column
fileName = 'model/tfidf_vectorizer.pkl'
with open(fileName,'rb') as f:
    tfidf_object = pickle.load(f)

#load average price by hull material, fuel type, and category
fileName = 'model/price_by_hull_material.pkl'
with open(fileName,'rb') as f:
    price_by_hull_material = pickle.load(f)

fileName = 'model/price_by_fuel_type.pkl'
with open(fileName,'rb') as f:
    price_by_fuel_type = pickle.load(f)

fileName = 'model/price_by_category.pkl'
with open(fileName,'rb') as f:
    price_by_category = pickle.load(f)

@ShipPricePredApp.get('/')
def index():
    #default return message
    return{'message': 'use /predict to make shipping prediction'}

@ShipPricePredApp.post('/predict')
def predict_price(data: ShipPricePred):
    #convert JSON payload to dictionary
    data = data.dict()
    
    #use helper function to clean data and create features
    sample = helper.build_pred_dict(data, tfidf_object, price_by_hull_material, price_by_fuel_type, price_by_category)
    
    #predict the data
    prediction = reconstructed_model.predict(sample)
    prediction = prediction[0][0]
    data = {"predicted_price": str(prediction)}
    
    return JSONResponse(content=data)



if __name__ == '__main__':
    uvicorn.run("main:ShipPricePredApp",host='127.0.0.1', port=8005, reload=True, workers=3)


