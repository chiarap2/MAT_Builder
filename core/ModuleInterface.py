from __future__ import annotations
from abc import ABC, abstractmethod


class ModuleInterface(ABC) :
    """
    The 'ModuleInterface' abstract class specifies the methods that a module must implement.
    """

   ### STATIC FIELDS ###
    
    id_class = 'ModuleInterface'
    
    
 
    ### INTERFACE METHODS ###
        
    @abstractmethod
    def execute(self, dic_params : dict) -> bool:
        """
        This method executes the internal task logic of a module.

        Parameters
        ----------
        dic_params: dict
            Dictionary that provides the input required by the module to execute its internal task logic.
            The dictionary contains (key,value) pairs, where key is the name of a specific input parameter and value
            the value passed for that parameter.
        """

        pass
        
    @abstractmethod
    def get_results(self) -> dict :
        """
        This method returns the results computed by the module.

        Returns
        -------
        dic_params: dict
            Dictionary that provides the output computed by the module after executing its internal task logic.
            The dictionary contains (key,value) pairs, where key is the name of a specific output parameter and value
            the value that has been computed for that parameter.
        """

        pass

    @abstractmethod
    def get_params_input(self) -> list[str] :
        """
        This method returns the list of input parameters that must be provided to the module.

        Returns
        -------
        dic_params: list[str]
            List containing the strings of input parameters that must be passed as keys in the dictionary required by the 'execute' method.
        """

        pass

    @abstractmethod
    def get_params_output(self) -> list[str] :
        """
        This method returns the list of output parameters returned by the module.

        Returns
        -------
        dic_params: list[str]
            List containing the strings of output parameters passed as keys in the dictionary returned by the 'get_results' method.
        """

        pass

    @abstractmethod
    def reset_state(self) :
        """
        This method resets the internal state of the module.
        """

        pass