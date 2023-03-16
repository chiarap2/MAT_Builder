import uvicorn
from fastapi import FastAPI

from core.modules import API_Preprocessing, API_Segmentation, API_Enrichment


# Setup the API services corresponding to the semantic enrichment processor functionalities...
app = FastAPI()
api_preprocessing = API_Preprocessing(app)
api_segmentation = API_Segmentation(app)
api_enrichment = API_Enrichment(app)


# Run the uvicorn server (backend).
if __name__=="__main__":
    uvicorn.run("__main__:app", reload=True)
    # uvicorn.run("__main__:app", workers=4)
