import pandas as pd
import geopandas as gpd
import uuid
import os
import io

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Query, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

from core.APIModuleInterface import APIModuleInterface
from core.modules.Enrichment import Enrichment


class API_Enrichment(APIModuleInterface, Enrichment) :
    '''
    This class exposes the functionalities provided by the Enrichment module via an API endpoint.
    '''

    ### INNER CLASSES ###

    class Params(BaseModel):
        move_enrichment: bool = Field(Query(..., description="Boolean value specifying if the move segments should be augmented with the estimated transportation means."))
        max_dist: int = Field(Query(default=50, description="Maximum distance (in meters) beyond which a POI won't be associated with a stop segment."))
        dbscan_epsilon: int = Field(Query(default=50, description="DBSCAN parameter: used to cluster stop segments (and thus find systematic stops). It represents the distance (in meters) below which a stop can be included in an existing cluster."))
        systematic_threshold : int = Field(Query(default=5, description="DBSCAN parameter: minimum size a cluster of stops must have to be considered a cluster of systematic stops."))



    ### PROTECTED METHODS ###

    def _enrichment_callback(self, dic_params : dict) :

        task_id = dic_params['task_id']


        exe_ok = False
        try:
            exe_ok = self.execute(dic_params)
        except Exception as e:
            print(f"ERROR: some exception occurred: {e}")
        print(f"Execution outcome: {exe_ok}")


        # 1.1 - Now store to disk the DataFrame containing the results.
        if exe_ok:
            namefile = task_id + ".ttl"
            namefile_tmp = namefile + ".tmp"
            self._rdf_graph.serialize_graph(namefile_tmp)
            os.rename(namefile_tmp, namefile)
        else:
            namefile = task_id + ".error"
            io.open(namefile, 'w').close()


        # Reset the object state and remove the temporary file once it's been transmitted to the user.
        self.reset_state()



    ### PUBLIC CLASS CONSTRUCTOR ###

    def __init__(self, router : APIRouter) :

        # Execute the superclasses constructors.
        APIModuleInterface.__init__(self)
        Enrichment.__init__(self)


        # Set up the HTTP responses that can be sent to the requesters.
        responses_get = {200: {"content": {"application/octet-stream": {}},
                               "description": "Returns a RDF knowledge graph stored in Turtle (.ttl) format."},
                         404: {"content": {"application/json": {}},
                               "description": "Task is currently being processed or does not exist."},
                         500: {"content": {"application/json": {}},
                               "description": "Some error occurred during the semantic enrichment. Check the correctness" +
                                              " of the datasets provided in input."}}


        # Declare the path function operations associated with the API_Enrichment class.
        @router.get("/" + Enrichment.id_class + "/",
                    description="This path operation returns a RDF knowledge graph. The result is returned in a Turtle (ttl) file.",
                    response_class=FileResponse,
                    responses=responses_get)
        def enrich(background_tasks: BackgroundTasks,
                   task_id: str = Query(description="Task ID associated with a previously done POST request.")):

            # Now, find out whether the results are ready OR some error occurred OR the task is still being processed...
            # ...OR the task does not exist, and answer accordingly.

            # 1 - Task terminated successfully.
            namefile_ok = "./" + task_id + ".ttl"
            namefile_error = "./" + task_id + ".error"
            if os.path.isfile(namefile_ok):
                background_tasks.add_task(os.remove, namefile_ok)
                return FileResponse(path=namefile_ok,
                                    filename='results.ttl',
                                    media_type='application/octet-stream')

            # 2 - Task terminated with an error.
            elif os.path.isfile(namefile_error):
                background_tasks.add_task(os.remove, namefile_error)
                return JSONResponse(status_code=500,
                                    content={"message": f"Some error occurred during the processing of task {task_id}!"})

            # 2 - Task is still being processed or does not exist.
            else:
                return JSONResponse(status_code=404,
                                    content={"message": f"Task {task_id} is still being processed or does not exist!"})

        @router.post("/" + Enrichment.id_class + "/",
                     description="This path operation returns a task id that can be later used to retrieve" +
                                 " a RDF knowledge graph containing the enriched trajectories.  The result is returned as" +
                                 " Turtle (.ttl) file.",
                     responses=self.responses_post)
        def enrich(background_tasks : BackgroundTasks,
                   file_trajectories : UploadFile = File(description="pandas DataFrame, stored in Parquet format, containing the trajectory dataset."),
                   file_moves : UploadFile = File(description="pandas DataFrame, stored in Parquet format, containing the move segment dataset."),
                   file_stops: UploadFile = File(description="pandas DataFrame, stored in Parquet format, containing the stop segment dataset."),
                   file_pois: UploadFile = File(description="GeoPandas DataFrame, stored in Parquet format, containing the POI dataset. Its content must be structured according to the GeoPandas DataFrames downloaded from OpenStreetMap via the OSMnx library."),
                   file_social: UploadFile = File(description="pandas DataFrame, stored in Parquet format, containing the social media post dataset."),
                   file_weather: UploadFile = File(description="pandas DataFrame, stored in Parquet format, containing the historical weather dataset."),
                   params: API_Enrichment.Params = Depends()) -> API_Enrichment.ResponsePost :

            task_id = str(uuid.uuid4().hex)

            # Here we execute the internal code of the Enrichment subclass to do the trajectory enrichment...
            params_enrichment = {'task_id': task_id,
                                 'trajectories': pd.read_parquet(file_trajectories.file),
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
            print(f"Dictionary passed to the background enrichment: {params_enrichment}")

            # Here we set up the call to the enrichment method to be executed at a later time.
            background_tasks.add_task(self._enrichment_callback, params_enrichment)

            # Return the identifier of the task ID associated with the request.
            return API_Enrichment.ResponsePost(message=f"Task {task_id} queued!",
                                               task_id=task_id)

