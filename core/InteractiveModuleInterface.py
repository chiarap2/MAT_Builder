from abc import ABC, abstractmethod


class InteractiveModuleInterface(ABC) :
 
    ### INTERFACE METHODS ###
    
    @abstractmethod
    def register_prev_module(self, prev_module) :
        pass
    
    @abstractmethod   
    def populate_input_area(self) :
        pass
        
    @abstractmethod
    def get_input_and_execute(self, **kwargs) :
        pass
        
    @abstractmethod
    def get_results(self) -> dict:
        pass
        
    @abstractmethod
    def reset_state(self) :
        pass