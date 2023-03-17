import pandas as pd
import geopandas as gpd
import uuid
import os

from .Enrichment import Enrichment

from pydantic import BaseModel, Field
from fastapi import FastAPI, APIRouter, Depends, Query, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse


class API_Enrichment(Enrichment) :

    ### INNER CLASSES ###

    class Params(BaseModel):
        move_enrichment: bool = Field(Query(..., description="Boolean value specifying if the move segments should be augmented with the estimated transportation means."))
        max_dist: int = Field(Query(..., description="Maximum distance beyond which a POI won't be associated with a stop segment."))
        dbscan_epsilon: int = Field(Query(..., description="DBSCAN parameter: used to cluster stop segments (and thus find systematic stops). Determines the distance below which a stop can be included in an existing cluster."))
        systematic_threshold : int = Field(Query(..., description="DBSCAN parameter: minimum size a cluster of stops must have to be considered a cluster of systematic stops."))



    ### PUBLIC CLASS CONSTRUCTOR ###

    def __init__(self, router : APIRouter) :

        # Execute the superclass constructor.
        super().__init__()


        # Set up the HTTP responses that can be sent to the requesters.
        responses = {200: {"content": {"application/octet-stream": {}},
                           "description": "Return a RDF knowledge graph, stored in Turtle (ttl) format."},
                     500: {"description" : "Some error occurred during the enrichment. Check the correctness of the files being provided in input!"}}

        # Declare the path function operations associated with the API_Preprocessing class.
        @router.get("/" + Enrichment.id_class + "/",
                    description="This path operation returns a RDF knowledge graph. The result is returned in a Turtle (ttl) file.",
                    response_class=FileResponse,
                    responses=responses)
        def enrich(background_tasks : BackgroundTasks,
                   file_trajectories : UploadFile = File(description="pandas DataFrame, stored in Parquet format, containing the trajectory dataset."),
                   file_moves : UploadFile = File(description="pandas DataFrame, stored in Parquet format, containing the move segment dataset."),
                   file_stops: UploadFile = File(description="pandas DataFrame, stored in Parquet format, containing the stop segment dataset."),
                   file_pois: UploadFile = File(description="GeoPandas DataFrame, stored in Parquet format, containing the POI dataset. Its content must be structured according to the GeoPandas DataFrames downloaded from OpenStreetMap via the OSMnx library."),
                   file_social: UploadFile = File(description="pandas DataFrame, stored in Parquet format, containing the social media post dataset."),
                   file_weather: UploadFile = File(description="pandas DataFrame, stored in Parquet format, containing the historical weather dataset."),
                   params: API_Enrichment.Params = Depends()) -> FileResponse :

            # Here we execute the internal code of the Preprocessing subclass to do the trajectory preprocessing...
            params_enrichment = {'trajectories': pd.read_parquet(file_trajectories.file),
                                 'moves': pd.read_parquet(file_moves.file),
                                 'move_enrichment': params.move_enrichment,
                                 'stops': pd.read_parquet(file_stops.file),
                                 'poi_place': 'Rome, Italy',  # IGNORED, if path_poi is not None.
                                 'poi_categories': None,  # ['amenity'],  # IGNORED, if path_poi is not None.
                                 'path_poi': gpd.read_parquet(file_pois.file),
                                 'max_dist': params.max_dist,
                                 'dbscan_epsilon': params.dbscan_epsilon,
                                 'systematic_threshold': params.systematic_threshold,
                                 'social_enrichment': pd.read_parquet(file_social.file),
                                 "weather_enrichment": pd.read_parquet(file_weather.file),
                                 'create_rdf': True}
            self.execute(params_enrichment)


            # Now create a temporary file on disk, and instruct FASTAPI to delete the file once the function has terminated.
            namefile = str(uuid.uuid4()) + ".ttl"
            self._rdf_graph.serialize_graph(namefile)

            # Reset the object state and remove the temporary file once it's been transmitted to the user.
            self.reset_state()
            background_tasks.add_task(os.remove, namefile)

            # Return the response (will be a file).
            return FileResponse(path = namefile, filename = 'results.ttl')
