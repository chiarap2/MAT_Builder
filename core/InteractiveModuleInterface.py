from abc import ABC, abstractmethod


class InteractiveModuleInterface(ABC) :
 
    ### INTERFACE METHODS ###

    @abstractmethod
    def get_dependencies(self) -> list['InteractiveModuleInterface']:
        pass
    
    @abstractmethod
    def register_modules(self, list_modules : list['InteractiveModuleInterface']):
        pass
    
    @abstractmethod   
    def populate_input_area(self) :
        pass
        
    @abstractmethod
    def get_input_and_execute_task(self, **kwargs) :
        pass
        
    @abstractmethod
    def get_results(self) -> dict:
        pass
        
    @abstractmethod
    def reset_state(self) :
        pass