import pandas as pd
import uuid

from pydantic import BaseModel, Field
from fastapi import APIRouter, UploadFile, File, Query, Depends, BackgroundTasks
from fastapi.responses import Response, JSONResponse

from core.APIModuleInterface import APIModuleInterface
from .Segmentation import Segmentation


class API_Segmentation(APIModuleInterface, Segmentation) :
    '''
    This class exposes the functionalities provided by the Segmentation module via an API endpoint.
    '''

    ### INNER CLASSES ###

    class Params(BaseModel):
        min_duration_stop: int = Field(Query(..., description="Minimum duration of a stop (in minutes)."))
        max_stop_radius: float = Field(Query(..., description="Maximum radius a stop can have (in kilometers)"))
        token: str = Field(Query(..., description="Token sent from the client"))

    class Results(BaseModel):
        stops: dict = Field(description="pandas DataFrame (translated in JSON) containing the stop segments.")
        moves: dict = Field(description="pandas DataFrame (translated in JSON) containing the move segments.")


    def _segment_callback(self, dic_params : dict):

        task_id = dic_params['task_id']

        # Here we execute the internal code of the Preprocessing subclass to do the trajectory preprocessing...
        params_segmentation = {'trajectories': pd.read_parquet(dic_params['trajectories'].file),
                               'duration': dic_params['duration'],
                               'radius': dic_params['radius']}


        print(f"Background segmentation!")
        exe_ok = False
        try:
            exe_ok = self.execute(params_segmentation)
        except Exception as e:
            print(f"ERROR: some exception occurred: {e}")
        print(f"Execution outcome: {exe_ok}")


        if exe_ok :
            # Construct the (dict) response, containing the stops and moves dataframes (will be automatically translated to JSON by FastAPI).
            results = {'stops': self.stops.to_dict(),
                       'moves': self.moves.to_dict()}
            self.task_results[task_id] = results
            self.task_status[task_id] = API_Segmentation.TaskStatus.OK
        else :
            self.task_status[task_id] = API_Segmentation.TaskStatus.ERROR

        # Reset the object internal state.
        self.reset_state()



    ### PUBLIC CLASS CONSTRUCTOR ###

    def __init__(self, router : APIRouter):

        # Execute the superclass constructor.
        super().__init__()


        # Dictionary used to hold the results of the tasks.
        self.task_results = {}


        # Set up the HTTP responses that can be sent to the requesters.
        responses_get = {200: {"content": {"application/octet-stream": {}},
                               "description": "Returns two pandas DataFrames containing the stop and move segments, translated into JSON."},
                         204: {"description": "The task is still being processed and thus the results are not available yet."},
                         404: {"content": {"application/json": {}},
                               "description": "Task does not exist."},
                         500: {"content": {"application/json": {}},
                               "description": "Some error occurred during the segmentation. Check the correctness of the trajectory dataset being provided in input."}}

        # Declare the path function operations associated with the API_Preprocessing class.
        @router.get("/" + Segmentation.id_class + "/",
                    description="If the task execution ended succesfully, the operation returns the pandas DataFrames of the stops" +
                                " and the moves translated into the JSON format.",
                    response_model=API_Segmentation.Results,
                    responses=responses_get)
        def segment(task_id : str = Query(description="Task ID associated with a previously done POST request."),
                    token : str = Query(description="Token sent from the client.")) :

            # Now, find out whether the results are ready OR some error occurred OR the task is still being processed...
            # ...OR the task does not exist, and answer accordingly.

            # 1 - task does not exist.
            if task_id not in self.task_status:
                return JSONResponse(status_code=404,
                                    content={"message": f"Task {task_id} does not exist!"})

            # 2 - Task terminated successfully.
            elif self.task_status[task_id] == API_Segmentation.TaskStatus.OK:
                results = self.task_results[task_id]
                del self.task_status[task_id]
                del self.task_results[task_id]
                return results

            # 3 - Task terminated with an error.
            elif self.task_status[task_id] == API_Segmentation.TaskStatus.ERROR:
                del self.task_status[task_id]
                return JSONResponse(status_code=500,
                                    content={"message": f"Some error occurred during the processing of task {task_id}!"})

            # 2 - Task is still being processed.
            else:
                return Response(status_code=204)


        @router.post("/" + Segmentation.id_class + "/",
                     description="This path operation initiates a task that segments a dataset of trajectories into stop and move segments.",
                     responses=self.responses_post)
        def segment(background_tasks: BackgroundTasks,
                    file_trajectories: UploadFile = File(description="pandas DataFrame, stored in a Parquet file, containing a dataset of trajectories."),
                    params: API_Segmentation.Params = Depends()) -> API_Segmentation.ResponsePost:

            # Here we set up the parameters needed by the background preprocessing method.
            task_id = str(uuid.uuid4().hex)
            params_segmentation = {'task_id': task_id,
                                   'trajectories': file_trajectories,
                                   'duration': params.min_duration_stop,
                                   'radius': params.max_stop_radius}
            print(f"Dictionary passed to the background segmentation: {params_segmentation}")

            # Here we set up the call to the preprocessing method to be executed at a later time.
            background_tasks.add_task(self._segment_callback, params_segmentation)

            # Keep track of this task status.
            self.task_status[task_id] = API_Segmentation.TaskStatus.WAITING

            # Return the identifier of the task ID associated with the request.
            return API_Segmentation.ResponsePost(message=f"Task {task_id} queued!",
                                                 task_id=task_id)
