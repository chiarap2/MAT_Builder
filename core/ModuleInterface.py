from abc import ABC, abstractmethod


class ModuleInterface(ABC) :
 
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