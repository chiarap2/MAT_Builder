import pandas as pd

import skmob
from skmob.preprocessing import filtering, compression
from ptrail.core.TrajectoryDF import PTRAILDataFrame

from core.ModuleInterface import ModuleInterface


class Preprocessing(ModuleInterface) :
    '''
    This class models the preprocessing module. More specifically, an instance of this class takes in input a dataset of trajectories and:

    1) Filters out the outliers from the trajectories, i.e., samples which have an anomalous speed.
    2) removes the trajectories that have a number of samples below a specified threshold.
    3) If requested by the user, compresses the trajectories obtained after applying the steps 1 and 2.
    '''

    ### CLASS PUBLIC STATIC FIELDS ###

    id_class = 'Preprocessing'



    ### PUBLIC CLASS CONSTRUCTOR ###

    def __init__(self) :

        print(f"Executing constructor of class {self.id_class}!")
        self.reset_state()



    ### PUBLIC CLASS METHODS ###

    def core(self) -> bool :

        self._results = None
        gdf = self._trajectories.copy()

        # ## PREPROCESSING

        # eliminate trajectories with a number of points lower than num_point
        print(f"Filtering trajectories with less than {self._num_point} samples...")
        gdf = gdf[gdf['traj_id'].groupby(gdf['traj_id']).transform('size') >= self._num_point]

        # convert GeoDataFrame into pandas DataFrame
        # df = pd.DataFrame(gdf)

        # now create a TrajDataFrame from the pandas DataFrame
        tdf = skmob.TrajDataFrame(gdf, latitude = 'lat', longitude = 'lon',
                                  datetime = 'time', user_id = 'user', trajectory_id = 'traj_id')

        print("Filtering out the outliers...")
        ftdf = filtering.filter(tdf, max_speed_kmh = self._kmh)

        print("Compressing the trajectories...")
        ctdf = compression.compress(ftdf, spatial_radius_km = 0.2) if self.compress else None

        self._results = ctdf if ctdf is not None else ftdf
        return True

    def output(self) :
        self._results.to_parquet(self.path_output)

    def execute(self, dic_params: dict) -> bool :
        """
        This method executes the task logic associated with the Preprocessing module.

        Parameters
        ----------
        dic_params : dict
            Dictionary that provides the input required by the module to execute its internal task logic.
            The dictionary contains (key,value) pairs, where key is the name of a specific input parameter and value
            the value passed for that input parameter.
            The input parameters that must be passed within 'dic_params' are:
                - 'trajectories': pandas DataFrame containing the trajectory dataset.
                - 'speed': float value specifying the speed beyond which a trajectory sample is considered an outlier to be removed.
                - 'n_points': int value specifying the number of samples below which a trajectory will be removed from the dataset.
                - 'compress': bool value specifying whether the trajectory dataset must be compressed or not. Such step is perfomed
                              after the outliers have been removed and the trjectories with few samples have been removed from the dataset.

        Returns
        -------
            execution_status : bool
                'True' if the execution went well, 'False' otherwise.
        """

        # Salva nei campi dell'istanza l'input passato
        self._trajectories = dic_params['trajectories']
        self._kmh = dic_params['speed']
        self._num_point = dic_params['n_points']
        self.compress = dic_params['compress']

        # Esegui il codice core dell'istanza.
        return self.core()

    def get_results(self) -> dict:
        return {'preprocessed_trajectories': self._results.copy() if self._results is not None else None}

    def get_params_input(self) -> list[str] :
        return ['trajectories', 'speed' 'n_points', 'compress']

    def get_params_output(self) -> list[str] :
        return list(self.get_results().keys())

    def reset_state(self) :
        self._trajectories = None
        self._num_point = None
        self._kmh = None
        self._results = None