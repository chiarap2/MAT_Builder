import geopandas as gpd
import pandas as pd
from typing import Optional

import skmob
from skmob.preprocessing import filtering, compression
from ptrail.core.TrajectoryDF import PTRAILDataFrame

from core.ModuleInterface import ModuleInterface


class Preprocessing(ModuleInterface) :

    ### PUBLIC CLASS CONSTRUCTOR ###

    def __init__(self) :
        self._df = None



    ### PUBLIC CLASS METHODS ###

    def core(self) -> bool :

        self._df = None

        if self.path[-3:] == 'csv':
            gdf = pd.read_csv(self.path)
        elif self.path[-7:] == 'parquet':
            gdf = pd.read_parquet(self.path)
        else:
            return False


        # ## PREPROCESSING

        # eliminate trajectories with a number of points lower than num_point
        grouped = gdf.groupby('traj_id')
        gdf = grouped.filter(lambda x: len(x) >= self.num_point)

        # convert GeoDataFrame into pandas DataFrame
        df = pd.DataFrame(gdf)

        # now create a TrajDataFrame from the pandas DataFrame
        tdf = skmob.TrajDataFrame(df, latitude = 'lat', longitude = 'lon',
                                  datetime = 'time', user_id = 'user', trajectory_id = 'traj_id')
        ftdf = filtering.filter(tdf, max_speed_kmh = self.kmh)
        ctdf = compression.compress(ftdf, spatial_radius_km = 0.2)

        self._df = ctdf
        return True

    def output(self) :
        self._df.to_parquet(self.path_output)

    def execute(self, dic_params: dict) -> bool :

        # Salva nei campi dell'istanza l'input passato
        self.path = dic_params['path']
        self.kmh = dic_params['speed']
        self.num_point = dic_params['n_points']

        # Esegui il codice core dell'istanza.
        return self.core()

    def get_results(self) -> dict:
        return {'preprocessed_trajectories': self._df.copy() if self._df is not None else None}

    def get_params_input(self) -> list[str] :
        return ['path', 'speed' 'n_points']

    def get_params_output(self) -> list[str] :
        return ['preprocessed_trajectories']

    def reset_state(self) :
        self._df = None