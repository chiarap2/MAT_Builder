from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class APIModuleInterface(ABC):

    ### INTERNAL CLASSES ###

    class ResponsePost(BaseModel):
        message: str = Field(description="Message returned by the server.")
        task_id: str = Field(description="Task ID associated with the request (if response is successful).")



    ### ABSTRACT CLASS CONSTRUCTOR ###

    def __init__(self):
        self.responses_post = {200: {"description": "A task has been successfully queued."}}