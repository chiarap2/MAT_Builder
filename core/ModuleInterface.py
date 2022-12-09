from __future__ import annotations
from abc import ABC, abstractmethod


class ModuleInterface(ABC) :

   ### STATIC FIELDS ###
    
    id_class = 'ModuleInterface'
    
    
 
    ### INTERFACE METHODS ###
    
    @abstractmethod
    def register_prev_module(self, prev_module : ModuleInterface) :
        pass
    
    @abstractmethod   
    def populate_input_area(self) :
        pass
        
    @abstractmethod
    def get_input_and_execute(self) :
        pass
        
    @abstractmethod
    def get_results(self) :
        pass
        
    @abstractmethod
    def reset_state(self) :
        pass