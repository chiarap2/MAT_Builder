import pandas as pd
import uuid
import os

from pydantic import BaseModel, Field
from fastapi import FastAPI, APIRouter, Depends, Query, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse

from .Preprocessing import Preprocessing


class API_Preprocessing(Preprocessing) :
    '''
    This class exposes the functionalities provided by the Preprocessing module via an API endpoint.
    '''

    ### INNER CLASSES ###

    class Params(BaseModel):
        min_num_samples: int = Field(Query(..., description="Minimum number of samples a trajectory must have to be considered."))
        max_speed: float = Field(Query(..., description="Maximum speed a sample can have in a trajectory (in km/h)."))
        compress_trajectories: bool = Field(Query(..., description="Boolean value determining whether the trajectories should be compressed. This can likely speed up subsequent enrichment steps."))



    ### PUBLIC CLASS CONSTRUCTOR ###

    def __init__(self, router : APIRouter):

        # Execute the superclass constructor.
        super().__init__()


        # Set up the HTTP responses that can be sent to the requesters.
        responses = {200: {"content": {"application/octet-stream": {}},
                           "description": "Return a pandas DataFrame, stored in Parquet, containing the preprocessed trajectory dataset."},
                     500: {"description" : "Some error occurred during the preprocessing. Check the correctness of the trajectory dataset being passed!"}}

        # Declare the path function operations associated with the API_Preprocessing class.
        @router.get("/" + Preprocessing.id_class + "/",
                    description="This path operation returns a dataset of preprocessed trajectories. The result is returned as a pandas DataFrame, stored in a Parquet file.",
                    response_class=FileResponse,
                    responses=responses)
        def preprocess(background_tasks : BackgroundTasks,
                       file_trajectories: UploadFile = File(description="pandas DataFrame, stored in a Parquet file, containing a dataset of trajectories."),
                       params: API_Preprocessing.Params = Depends(API_Preprocessing.Params)) :

            # Here we execute the internal code of the Preprocessing subclass to do the trajectory preprocessing...
            params_preprocessing = {'trajectories': pd.read_parquet(file_trajectories.file),
                                    'speed': params.max_speed,
                                    'n_points': params.min_num_samples,
                                    'compress': params.compress_trajectories}
            self.execute(params_preprocessing)

            # Now create a temporary file on disk, and instruct FASTAPI to delete the file once the function has terminated.
            namefile = str(uuid.uuid4()) + ".parquet"
            self._results.to_parquet(namefile)

            # Reset the object internal state and delete the file once it has been transmitted in the response.
            background_tasks.add_task(os.remove, namefile)
            self.reset_state()

            # Return the response (will be a file).
            return FileResponse(path = namefile, filename = 'preprocessed_trajectories.parquet', media_type='application/octet-stream')
