import geopandas as gpd
import pandas as pd

import skmob
from skmob.preprocessing import filtering, compression
from ptrail.core.TrajectoryDF import PTRAILDataFrame

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL

import plotly.express as px

from core.Pipeline import Pipeline
from core.ModuleInterface import ModuleInterface


class Preprocessing(ModuleInterface):

    ### PUBLIC CLASS CONSTRUCTOR ###

    def __init__(self):
        self.df = None


    ### PUBLIC CLASS METHODS ###

    def core(self) -> None :

        self.df = None

        if self.path[-3:] == 'csv':
            gdf = pd.read_csv(self.path)
        elif self.path[-7:] == 'parquet':
            gdf = pd.read_parquet(self.path)
        else:
            return


        # ## PREPROCESSING

        # eliminate trajectories with a number of points lower than num_point
        grouped = gdf.groupby('traj_id')
        gdf = grouped.filter(lambda x: len(x) >= self.num_point)

        # convert GeoDataFrame into pandas DataFrame
        df = pd.DataFrame(gdf)

        # now create a TrajDataFrame from the pandas DataFrame
        tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', datetime='time', user_id='user',
                                  trajectory_id='traj_id')
        ftdf = filtering.filter(tdf, max_speed_kmh=self.kmh)
        ctdf = compression.compress(ftdf, spatial_radius_km=0.2)

        self.df = ctdf

    def get_num_users(self) :
        return str(len(self.df.uid.unique()))

    def get_num_trajs(self):
        return str(len(self.df.tid.unique()))

    def output(self):
        self.df.to_parquet(self.path_output)

    def execute(self, dic_params: dict) :

        # Salva nei campi dell'istanza l'input passato
        self.path = dic_params['path']
        self.kmh = dic_params['speed']
        self.num_point = dic_params['n_points']

        # Esegui il codice core dell'istanza.
        self.core()

    def get_results(self) -> dict :
        return {'traj_preprocessed': self.df.copy()}

    def reset_state(self) :
        self.df = None