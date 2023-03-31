import pandas as pd
import uuid
import os

from pydantic import BaseModel, Field
from fastapi import Response, APIRouter, Depends, Query, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

from core.APIModuleInterface import APIModuleInterface
from .Preprocessing import Preprocessing


class API_Preprocessing(APIModuleInterface, Preprocessing) :
    '''
    This class exposes the functionalities provided by the Preprocessing module via an API endpoint.
    '''

    ### INNER CLASSES ###

    class Params(BaseModel):
        min_num_samples: int = Field(Query(..., description="Minimum number of samples a trajectory must have to be considered."))
        max_speed: float = Field(Query(..., description="Maximum speed a sample can have in a trajectory (km/h)."))
        compress_trajectories: bool = Field(Query(..., description="Boolean value determining whether the trajectories should be compressed. This can likely speed up subsequent enrichment steps."))
        token: str = Field(Query(..., description="Token sent from the client"))



    ### PROTECTED CLASS METHODS ###

    def _preprocess_callback(self, dic_params : dict):

        task_id = dic_params['task_id']

        print(f"Background preprocessing!")
        params_preprocessing = {'trajectories': pd.read_parquet(dic_params['trajectories'].file),
                                'speed': dic_params['speed'],
                                'n_points': dic_params['n_points'],
                                'compress': dic_params['compress']}
        exe_ok = False
        try:
            exe_ok = self.execute(params_preprocessing)
        except Exception as e:
            print(f"ERROR: some exception occurred: {e}")
        print(f"Execution outcome: {exe_ok}")


        # 1.1 - Now store to disk the DataFrame containing the results.
        if exe_ok :
            self.task_status[task_id] = API_Preprocessing.TaskStatus.OK
            namefile = task_id + ".parquet"
            self._results.to_parquet(namefile)
        else :
            self.task_status[task_id] = API_Preprocessing.TaskStatus.ERROR

        # Reset the object's internal state.
        self.reset_state()



    ### PUBLIC CLASS CONSTRUCTOR ###

    def __init__(self, router : APIRouter):

        # Execute the superclass constructor.
        super().__init__()


        # Set up the HTTP responses that can be sent to the requesters.
        responses_get = {200: {"content": {"application/octet-stream": {}},
                               "description": "Return a pandas DataFrame, stored in Parquet, containing the preprocessed trajectory dataset."},
                         204: {"content": {"application/json": {}},
                               "description": "The task is still being processed and thus the results are not available yet."},
                         404: {"content": {"application/json": {}},
                               "description": "Task does not exist."},
                         500: {"content": {"application/json": {}},
                               "description": "Some error occurred during the preprocessing. Check the correctness of the trajectory dataset being passed."}}


        # Declare the path function operations associated with the API_Preprocessing class.
        @router.get("/" + Preprocessing.id_class + "/",
                    description="This path operation returns a dataset of preprocessed trajectories." +
                                "If ready, the result is returned as a pandas DataFrame, stored in a Parquet file.",
                    response_class=FileResponse,
                    responses=responses_get)
        def preprocess(background_tasks : BackgroundTasks,
                       task_id : str = Query(description="Task ID associated with a previously done POST request."),
                       token : str = Query(description="Token sent from the client.")) :

            # Now, find out whether the results are ready OR some error occurred OR the task is still being processed...
            # ...OR the task does not exist, and answer accordingly.

            # 1 - task does not exist.
            if task_id not in self.task_status :
                return JSONResponse(status_code=404, content={"message": f"Task {task_id} does not exist!"})

            # 2 - Task terminated successfully.
            elif self.task_status[task_id] == API_Preprocessing.TaskStatus.OK :
                namefile = "./" + task_id + ".parquet"
                background_tasks.add_task(os.remove, namefile)
                del self.task_status[task_id]
                return FileResponse(path=namefile,
                                    filename='preprocessed_trajectories.parquet',
                                    media_type='application/octet-stream')

            # 3 - Task terminated with an error.
            elif self.task_status[task_id] == API_Preprocessing.TaskStatus.ERROR :
                del self.task_status[task_id]
                return JSONResponse(status_code=500,
                                    content={"message": f"Some error occurred during the processing of task {task_id}!"})

            # 2 - Task is still being processed.
            else :
                print("Caso attesa.")
                return Response(status_code=204)


        @router.post("/" + Preprocessing.id_class + "/",
                     description="This path operation returns a task id that can be later used to retrieve" +
                                 " a dataset of preprocessed trajectories. The result is returned as a pandas" +
                                 " DataFrame, stored in a Parquet file.",
                     responses=self.responses_post)
        def preprocess(background_tasks: BackgroundTasks,
                       file_trajectories: UploadFile = File(description="pandas DataFrame, stored in a Parquet file, containing a dataset of trajectories."),
                       params: API_Preprocessing.Params = Depends(API_Preprocessing.Params)) -> API_Preprocessing.ResponsePost :

            # Here we set up the parameters needed by the background preprocessing method.
            task_id = str(uuid.uuid4().hex)
            params_preprocessing = {'task_id' : task_id,
                                    'trajectories': file_trajectories,
                                    'speed': params.max_speed,
                                    'n_points': params.min_num_samples,
                                    'compress': params.compress_trajectories}
            print(f"Dictionary passed to the background preprocessing: {params_preprocessing}")

            # Here we set up the call to the preprocessing method to be executed at a later time.
            background_tasks.add_task(self._preprocess_callback, params_preprocessing)

            # Keep track of this task status.
            self.task_status[task_id] = API_Preprocessing.TaskStatus.WAITING

            # Return the identifier of the task ID associated with the request.
            return API_Preprocessing.ResponsePost(message = f"Task {task_id} queued!",
                                                  task_id = task_id)