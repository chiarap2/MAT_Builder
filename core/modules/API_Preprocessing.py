import pandas as pd
import uuid
import os

from pydantic import BaseModel
from fastapi import FastAPI, Depends, Query, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse

from .Preprocessing import Preprocessing


class API_Preprocessing(Preprocessing) :

    ### INNER CLASSES ###

    class Params(BaseModel):
        min_num_samples: int = Query(...)
        max_speed: float = Query(...)
        compress_trajectories: bool = Query(...)



    ### PUBLIC CLASS CONSTRUCTOR ###

    def __init__(self, app : FastAPI):

        # Execute the superclass constructor.
        super().__init__()

        # Declare the path function operations associated with the API_Preprocessing class.
        @app.get("/semantic_processor/" + Preprocessing.id_class + "/", response_class=FileResponse)
        def preprocess(background_tasks : BackgroundTasks,
                       file_trajectories: UploadFile,
                       params: API_Preprocessing.Params = Depends()) -> FileResponse :

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
            return FileResponse(path = namefile, filename = 'preprocessed_trajectories.parquet')
