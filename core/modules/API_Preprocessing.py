import pandas as pd
import uuid
import os

from .Preprocessing import Preprocessing

from fastapi import FastAPI, Form, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse


class API_Preprocessing(Preprocessing) :

    def __init__(self, app : FastAPI):

        # Execute the superclass constructor.
        super().__init__()

        # Declare the path function operations associated with the API_Preprocessing class.
        @app.get("/semantic_processor/" + self.id_class + "/")
        async def preprocess(background_tasks : BackgroundTasks,
                             file_trajectories : UploadFile,
                             min_num_samples : int = Form(),
                             max_speed : float = Form(),
                             compress_trajectories : bool = Form()) :

            # Here we execute the internal code of the Preprocessing subclass to do the trajectory preprocessing...
            params_preprocessing = {'trajectories': pd.read_parquet(file_trajectories.file),
                                    'speed': max_speed,
                                    'n_points': min_num_samples,
                                    'compress': compress_trajectories}
            self.execute(params_preprocessing)

            # Now create a temporary file on disk, and instruct FASTAPI to delete the file once the function has terminated.
            namefile = str(uuid.uuid4()) + ".parquet"
            self._results.to_parquet(namefile)
            background_tasks.add_task(os.remove, namefile)

            # Return the response (will be a file).
            return FileResponse(path = namefile, filename = 'preprocessed_trajectories.parquet')



