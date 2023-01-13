from __future__ import annotations
from abc import ABC, abstractmethod


class ModuleInterface(ABC) :

   ### STATIC FIELDS ###
    
    id_class = 'ModuleInterface'
    
    
 
    ### INTERFACE METHODS ###
        
    @abstractmethod
    def execute(self, dic_params : dict) :
        pass
        
    @abstractmethod
    def get_results(self) -> dict :
        pass

    @abstractmethod
    def reset_state(self) :
        pass