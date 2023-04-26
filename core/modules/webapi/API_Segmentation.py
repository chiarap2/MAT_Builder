import pandas as pd
import uuid
import os
import io
import json

from pydantic import BaseModel, Field
from fastapi import APIRouter, UploadFile, File, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from core.APIModuleInterface import APIModuleInterface
from core.modules.Segmentation import Segmentation


class API_Segmentation(APIModuleInterface, Segmentation) :
    '''
    This class exposes the functionalities provided by the Segmentation module via an API endpoint.
    '''

    ### INNER CLASSES ###

    class Params(BaseModel):
        min_duration_stop: int = Field(Query(..., default=10, description="Minimum duration of a stop (in minutes)."))
        max_stop_radius: float = Field(Query(..., default=0.2, description="Maximum radius a stop can have (in kilometers)"))

    class SegmentationResults(BaseModel):
        stops: dict = Field(description="pandas DataFrame (translated in JSON) containing the stop segments.")
        moves: dict = Field(description="pandas DataFrame (translated in JSON) containing the move segments.")


    def _segment_callback(self, dic_params : dict):

        task_id = dic_params['task_id']


        print(f"Background segmentation!")
        exe_ok = False
        try:
            exe_ok = self.execute(dic_params)
        except Exception as e:
            print(f"ERROR: some exception occurred: {e}")
        print(f"Execution outcome: {exe_ok}")


        if exe_ok :
            # Construct the (dict) response, containing the stops and moves dataframes (will be automatically translated to JSON by FastAPI).
            results = {'stops': self.stops.to_dict(),
                       'moves': self.moves.to_dict()}
            with open(task_id + '.json', 'w', encoding='utf-8') as out_file:
                json.dump(results, out_file, default=str, ensure_ascii=False)
        else :
            namefile = task_id + ".error"
            io.open(namefile, 'w').close()

        # Reset the object internal state.
        self.reset_state()



    ### PUBLIC CLASS CONSTRUCTOR ###

    def __init__(self, router : APIRouter):

        # Execute the superclasses constructors.
        APIModuleInterface.__init__(self)
        Segmentation.__init__(self)


        # Set up the HTTP responses that can be sent to the requesters.
        responses_get = {200: {"content": {"application/json": {}},
                               "description": "Returns the two pandas DataFrames containing the stop and move segments, in JSON format."},
                         404: {"content": {"application/json": {}},
                               "description": "Task is still being processed or does not exist."},
                         500: {"content": {"application/json": {}},
                               "description": "Some error occurred during the trajectory segmentation. Check the correctness of the" +
                                              "trajectory dataset being provided in input."}}

        # Declare the path function operations associated with the API_Preprocessing class.
        @router.get("/" + Segmentation.id_class + "/",
                    description="If the task execution ended successfully, the operation returns the pandas DataFrames of the stops" +
                                " and the moves translated into the JSON format.",
                    response_model=API_Segmentation.SegmentationResults,
                    responses=responses_get)
        def segment(background_tasks : BackgroundTasks,
                    task_id : str = Query(description="Task ID associated with a previously done POST request.")) :

            # Now, find out whether the results are ready OR some error occurred OR the task is still being processed...
            # ...OR the task does not exist, and answer accordingly.

            # 1 - Task terminated successfully.
            namefile_ok = "./" + task_id + ".json"
            namefile_error = "./" + task_id + ".error"
            if os.path.isfile(namefile_ok) :
                background_tasks.add_task(os.remove, namefile_ok)
                with open(namefile_ok, "r") as f:
                    results = json.loads(f.read())
                    return results

            # 2 - Task terminated with an error.
            elif os.path.isfile(namefile_error):
                background_tasks.add_task(os.remove, namefile_error)
                return JSONResponse(status_code=500,
                                    content={"message": f"Some error occurred during the processing of task {task_id}!"})

            # 3 - Task is still being processed.
            else:
                return JSONResponse(status_code=404,
                                    content={"message": f"Task {task_id} is still being processed or does not exist!"})


        @router.post("/" + Segmentation.id_class + "/",
                     description="This path operation initiates a task that segments a dataset of trajectories into stop and move segments.",
                     responses=self.responses_post)
        def segment(background_tasks: BackgroundTasks,
                    file_trajectories: UploadFile = File(description="pandas DataFrame, stored in a Parquet file, containing a dataset of trajectories."),
                    params: API_Segmentation.Params = Depends()) -> API_Segmentation.ResponsePost:

            # Here we set up the parameters needed by the background preprocessing method.
            task_id = str(uuid.uuid4().hex)
            params_segmentation = {'task_id': task_id,
                                   'trajectories': pd.read_parquet(file_trajectories.file),
                                   'duration': params.min_duration_stop,
                                   'radius': params.max_stop_radius}
            print(f"Dictionary passed to the background segmentation: {params_segmentation}")

            # Here we set up the call to the preprocessing method to be executed at a later time.
            background_tasks.add_task(self._segment_callback, params_segmentation)

            # Return the identifier of the task ID associated with the request.
            return API_Segmentation.ResponsePost(message=f"Task {task_id} queued!",
                                                 task_id=task_id)
