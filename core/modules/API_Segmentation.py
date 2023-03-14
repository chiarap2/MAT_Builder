import pandas as pd

from fastapi import FastAPI, Form, UploadFile

from .Segmentation import Segmentation


class API_Segmentation(Segmentation) :

    def __init__(self, app : FastAPI):

        # Execute the superclass constructor.
        super().__init__()

        # Declare the path function operations associated with the API_Preprocessing class.
        @app.get("/semantic_processor/" + self.id_class + "/")
        async def segment(file_trajectories : UploadFile,
                          min_duration_stop : int = Form(),
                          max_stop_radius : float = Form()) :

            # Here we execute the internal code of the Preprocessing subclass to do the trajectory preprocessing...
            params_segmentation = {'trajectories': pd.read_parquet(file_trajectories.file),
                                   'duration': min_duration_stop,
                                   'radius': max_stop_radius}

            self.execute(params_segmentation)


            # Return a JSON response containing the stops and moves dataframes (translated to JSON).

            # 2 - we can then return a dict containing the stops and moves dataframes.
            return {'stops' : self.stops.to_dict(),
                    'moves' : self.moves.to_dict()}