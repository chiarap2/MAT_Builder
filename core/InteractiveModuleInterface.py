from abc import ABC, abstractmethod


class InteractiveModuleInterface(ABC) :
    """
    This abstract class specifies the methods that an interactive module must implement.
    """
 
    ### INTERFACE METHODS ###

    @abstractmethod
    def get_dependencies(self) -> list['InteractiveModuleInterface']:
        """
        This method returns the list of type references associated with the modules from whose output the module depends.

        Returns
        -------
        list of type references : list
            List of type references
        """

        pass
    
    @abstractmethod
    def register_modules(self, list_modules : list['InteractiveModuleInterface']):
        """
        This method registers the instances of the modules from whose output the module depends.

        Parameters
        -------
        list_modules : list[InteractiveModuleInterface]
            List containing the references to the instances of the modules from whose output the module depends.
        """

        pass
    
    @abstractmethod   
    def populate_input_area(self) :
        """
        This method is in charge of populating the UI input area.
        """

        pass
        
    @abstractmethod
    def get_input_and_execute_task(self, **kwargs) :
        """
        This method is in charge of getting the input provided by the user via the UI, and then
        execute the task logic associated with the module.

        Parameters
        -------
        **kwargs :
            Additional keyword arguments, corresponding to the input parameters taken by the specific module.
        """

        pass
        
    @abstractmethod
    def get_results(self) -> dict:
        """
        This method returns the results computed by the module.

        Returns
        -------
        results : dict
            Dictionary containing the various results.
        """

        pass
        
    @abstractmethod
    def reset_state(self) :
        """
        This method resets the internal state of the module.
        """

        pass