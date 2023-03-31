from abc import ABC, abstractmethod
from enum import Enum
from pydantic import BaseModel, Field


class APIModuleInterface(ABC):

    ### INTERNAL CLASSES ###

    class TaskStatus(Enum):
        WAITING = 0
        OK = 1
        ERROR = 2

    class ResponsePost(BaseModel):
        message: str = Field(description="Message returned by the server.")
        task_id: str = Field(description="Task ID associated with the request (if response is successful).")



    ### ABSTRACT CLASS CONSTRUCTOR ###

    def __init__(self):
        self.task_status = {}
        self.responses_post = {200: {"description": "A task has been successfully queued."}}