from fastapi import FastAPI
from core.modules import API_Preprocessing, API_Segmentation


app = FastAPI()

api_preprocessing = API_Preprocessing(app)
api_segmentation = API_Segmentation(app)