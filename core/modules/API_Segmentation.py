import pandas as pd

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, Query, Depends

from .Segmentation import Segmentation


class API_Segmentation(Segmentation) :

    ### INNER CLASSES ###

    class Params(BaseModel):
        min_duration_stop: int = Query(...)
        max_stop_radius: float = Query(...)

    class Results(BaseModel):
        stops : dict
        moves : dict



    ### PUBLIC CLASS CONSTRUCTOR ###

    def __init__(self, app : FastAPI):

        # Execute the superclass constructor.
        super().__init__()

        # Declare the path function operations associated with the API_Preprocessing class.
        @app.get("/semantic_processor/" + Segmentation.id_class + "/")
        def segment(file_trajectories : UploadFile,
                    params: API_Segmentation.Params = Depends()) -> API_Segmentation.Results:

            # Here we execute the internal code of the Preprocessing subclass to do the trajectory preprocessing...
            params_segmentation = {'trajectories': pd.read_parquet(file_trajectories.file),
                                   'duration': params.min_duration_stop,
                                   'radius': params.max_stop_radius}

            self.execute(params_segmentation)


            # Construct the (dict) response, containing the stops and moves dataframes (will be automatically translated to JSON by FastAPI).
            results = {'stops' : self.stops.to_dict(),
                       'moves' : self.moves.to_dict()}

            # Reset the object internal state.
            self.reset_state()

            # Return the results.
            return results
