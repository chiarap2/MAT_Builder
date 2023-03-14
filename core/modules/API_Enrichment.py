import pandas as pd
import geopandas as gpd
import uuid
import os

from .Enrichment import Enrichment

from fastapi import FastAPI, Form, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse


class API_Enrichment(Enrichment) :

    def __init__(self, app : FastAPI):

        # Execute the superclass constructor.
        super().__init__()

        # Declare the path function operations associated with the API_Preprocessing class.
        @app.get("/semantic_processor/" + self.id_class + "/")
        async def enrich(background_tasks : BackgroundTasks,
                         file_trajectories : UploadFile,
                         file_moves : UploadFile,
                         file_stops: UploadFile,
                         file_pois: UploadFile,
                         file_social: UploadFile,
                         file_weather: UploadFile,
                         move_enrichment : bool = Form(),
                         max_dist : int = Form(),
                         dbscan_epsilon : int = Form(),
                         systematic_threshold : int = Form()) :

            # Here we execute the internal code of the Preprocessing subclass to do the trajectory preprocessing...
            params_enrichment = {'trajectories': pd.read_parquet(file_trajectories.file),
                                 'moves': pd.read_parquet(file_moves.file),
                                 'move_enrichment': move_enrichment,
                                 'stops': pd.read_parquet(file_stops.file),
                                 'poi_place': 'Rome, Italy',  # IGNORED, if path_poi is not None.
                                 'poi_categories': None,  # ['amenity'],  # IGNORED, if path_poi is not None.
                                 'path_poi': gpd.read_parquet(file_pois.file),
                                 'max_dist': max_dist,
                                 'dbscan_epsilon': dbscan_epsilon,
                                 'systematic_threshold': systematic_threshold,
                                 'social_enrichment': pd.read_parquet(file_social.file),
                                 "weather_enrichment": pd.read_parquet(file_weather.file),
                                 'create_rdf': True}
            self.execute(params_enrichment)

            # Now create a temporary file on disk, and instruct FASTAPI to delete the file once the function has terminated.
            namefile = str(uuid.uuid4()) + ".ttl"
            self._rdf_graph.serialize_graph(namefile)
            background_tasks.add_task(os.remove, namefile)

            # Return the response (will be a file).
            return FileResponse(path = namefile, filename = 'results.ttl')
