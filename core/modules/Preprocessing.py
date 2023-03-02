import pandas as pd

import skmob
from skmob.preprocessing import filtering, compression
from ptrail.core.TrajectoryDF import PTRAILDataFrame

from core.ModuleInterface import ModuleInterface


class Preprocessing(ModuleInterface) :

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