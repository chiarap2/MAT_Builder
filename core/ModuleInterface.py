from __future__ import annotations
from abc import ABC, abstractmethod


class ModuleInterface(ABC) :

   ### STATIC FIELDS ###
    
    id_class = 'ModuleInterface'
    
    
 
    ### INTERFACE METHODS ###
        
    @abstractmethod
    def execute(self, dic_params : dict) -> bool:
        pass
        
    @abstractmethod
    def get_results(self) -> dict :
        pass

    @abstractmethod
    def get_params_input(self) -> list[str] :
        pass

    @abstractmethod
    def get_params_output(self) -> list[str] :
        pass

    @abstractmethod
    def reset_state(self) :
        pass