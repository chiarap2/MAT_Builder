import uvicorn
from fastapi import FastAPI, APIRouter

from core import API_Preprocessing, API_Segmentation, API_Enrichment


# Setup of the semantic enrichment processor web API.
app = FastAPI(title="Semantic Enrichment Processor API",
              description="MobiDataLab semantic enrichment API",
              contact={"url": "https://github.com/MobiDataLab/mdl-semantic-enrichment"},
              version='1.0.2')

# Setup of the various web API endpoints.
prefix_router = APIRouter(prefix="/semantic")
api_preprocessing = API_Preprocessing(prefix_router)
api_segmentation = API_Segmentation(prefix_router)
api_enrichment = API_Enrichment(prefix_router)
app.include_router(prefix_router)


# Run the uvicorn server (backend).
if __name__=="__main__":
    # uvicorn.run("__main__:app", reload=True)
    uvicorn.run("__main__:app", host="0.0.0.0", workers=5)
