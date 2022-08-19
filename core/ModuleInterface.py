from abc import ABC, abstractmethod


class ModuleInterface(ABC) :
 
    ### INTERFACE METHODS ###
    
    @abstractmethod
    def register_prev_module(self, prev_module) :
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