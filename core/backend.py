import geopandas as gpd
import shapely
import pandas as pd
import numpy as np
import skmob
from skmob.preprocessing import filtering
from skmob.preprocessing import detection
from skmob.preprocessing import compression
from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.kinematic_features import KinematicFeatures as spatial
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

shapely.speedups.disable()

class demo():

    '''
    Class `demo` is a super class containing one subclass per module (in our case `preprocessing`, `segmentation`, `enrichment`)
    '''

    def __init__(self):
        '''
        Initialize `subclasses` that is a list of all subclasses
        '''
        subclasses = self.__subclasses__()

class Preprocessing(demo):
    '''
    `preprocessing` module
    '''
    pass

class Segmentation(demo):
    '''
    `segmentation` module
    '''
    pass

class Enrichment(demo):
    '''
    `enrichment` module
    '''
    pass

class preprocessing1(Preprocessing):
    '''
    `preprocessing1` is a subclass of `Preprocessing` to preprocess trajectories and allows users to:
    1) remove trajectories with a few number of points
    2) remove outliers
    3) compress trajectories
    '''
    pass

    df = gpd.GeoDataFrame()
    origins = gpd.GeoDataFrame()
    destinations = gpd.GeoDataFrame()

    def __init__(self, list_):

        self.path = list_[0]
        self.num_point = list_[2]
        self.kmh = list_[1]

    def core(self):

        #gdf = pd.read_csv(self.path)
        gdf = pd.read_parquet(self.path)
        print(gdf.head())
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

        #gdf = gpd.GeoDataFrame(ctdf)

        #gb = gdf.groupby(['tid'], as_index=False)
        #origins = gb.first()
        #destinations = gb.last()

        #self.origins = origins
        #self.destinations = destinations

        #if gdf.crs is None:
        #    gdf.set_crs('epsg:4326',inplace=True)
        #    gdf.to_crs('epsg:3857',inplace=True)
        #else:
        #    gdf.to_crs('epsg:3857',inplace=True)

        self.df = ctdf

    def get_num_users(self):

        return str(len(self.df.uid.unique()))

    def get_num_trajs(self):

        return str(len(self.df.tid.unique()))

    def output(self):

        self.df.to_parquet('data/temp_dataset/traj_cleaned.parquet')

    #def graphic_interface(self):

class preprocessing2(Preprocessing):
    '''
    `preprocessing2` is a subclass of `Preprocessing` to preprocess trajectories and allows users to:
    1) remove outliers
    2) compress trajectories
    '''
    pass

    df = gpd.GeoDataFrame()

    def __init__(self,list_):

        self.path = list_[0]
        self.kmh = list_[1]

    def core(self):

        df = pd.read_csv(self.path)

        # ## PREPROCESSING

        # create a TrajDataFrame from the pandas DataFrame
        tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', datetime='time', user_id='track_fid',
                                  trajectory_id='traj_id')
        ftdf = filtering.filter(tdf, max_speed_kmh=self.kmh)
        ctdf = compression.compress(ftdf, spatial_radius_km=0.2)


        gdf = gpd.GeoDataFrame(ctdf)

        if gdf.crs is None:
            gdf.set_crs('epsg:4326',inplace=True)
            gdf.to_crs('epsg:3857',inplace=True)
        else:
            gdf.to_crs('epsg:3857',inplace=True)

        self.df = gdf

    def output(self):

        self.df.to_parquet('data/temp_dataset/traj_cleaned.parquet')


class stops_and_moves(Segmentation):

    '''
    `stops_and_moves` is a subclass of `Segmentation` to detect stop points and moves.
    '''

    pass

    stops = pd.DataFrame()
    moves = pd.DataFrame()
    preprocessed_trajs = pd.DataFrame()

    def __init__(self, list_):
        
        self.minutes = list_[0]
        self.radius = list_[1]
        self.preprocessed_trajs = pd.read_parquet('data/temp_dataset/traj_cleaned.parquet')

    def core(self):
        
        # read preprocessed dataframe
        tdf = skmob.TrajDataFrame(self.preprocessed_trajs)

        # add speed, acceleration, distance info using PTRAIL
        #df = PTRAILDataFrame(tdf, latitude='lat', longitude='lng', datetime='datetime', traj_id='tid')
        #speed = spatial.create_speed_column(df)
        #bearing = spatial.create_bearing_rate_column(speed)
        #acceleration = spatial.create_jerk_column(bearing)

        # stop detection
        stdf = detection.stops(tdf, stop_radius_factor=0.5, minutes_for_a_stop=self.minutes, spatial_radius_km=self.radius, leaving_time=True)
        self.stops = stdf
        # save stops
        stdf.to_parquet('data/temp_dataset/stops.parquet')

        # move detection
        trajs = tdf.copy()
        starts = stdf.copy()
        ends = stdf.copy()

        trajs.set_index(['tid','datetime'],inplace=True)
        starts.set_index(['tid','datetime'],inplace=True)
        ends.set_index(['tid','leaving_datetime'],inplace=True)

        traj_ids = trajs.index
        start_ids = starts.index
        end_ids = ends.index

        # some datetime into stdf are approximated. In order to retrieve moves, we have to check the exact datime into 
        # trajectory dataframe
        # we use `isin()` method to reduce time computation
        traj_df = pd.DataFrame(traj_ids, columns=['trajs'])
        start_df = pd.DataFrame(start_ids, columns=['start'])
        end_df = pd.DataFrame(end_ids, columns=['end'])

        start_df['is_in_traj'] = start_df['start'].isin(traj_df['trajs'])
        end_df['is_in_traj'] = end_df['end'].isin(traj_df['trajs'])

        start_df['end'] = end_df['end']
        start_df['is_in_traj_end'] = end_df['is_in_traj']

        # remove stops which aren't into tdf
        start_df = start_df[(start_df['is_in_traj']!=False)|(start_df['is_in_traj_end']!=False)]

        

        # save index of incomplete stops and convert them into MultiIndex
        incomplete_end = start_df['end'][(start_df['is_in_traj']==False)&(start_df['is_in_traj_end']==True)] 
        incomplete_start = start_df['start'][(start_df['is_in_traj']==True)&(start_df['is_in_traj_end']==False)]

        if not incomplete_end.empty:
            incomplete_end = pd.MultiIndex.from_tuples(incomplete_end)

        if not incomplete_start.empty:
            incomplete_start = pd.MultiIndex.from_tuples(incomplete_start)

        # save complete index
        start_df = start_df[(start_df['is_in_traj']==True)&(start_df['is_in_traj_end']==True)] 
        
        new_start = pd.MultiIndex.from_tuples(start_df['start'])
        new_end = pd.MultiIndex.from_tuples(start_df['end'])
        new_start.set_names(['tid','datetime'],inplace=True)
        new_end.set_names(['tid','datetime'],inplace=True)
        
        # set start and end of stops (using two columns in order to avoid overlaps)
        trajs['start_stop'] = np.nan
        trajs['start_stop'].loc[new_start] = 1
        trajs['end_stop'] = np.nan
        trajs['end_stop'].loc[new_end] = 1  

        trajs.reset_index(inplace=True)
        start_idx = trajs[trajs['start_stop']==1].index.to_list()
        end_idx = trajs[trajs['end_stop']==1].index.to_list()

        # set incomplete index
        starts_ = [traj_ids.get_loc(e).start - 1 for e in incomplete_end]
        ends_ = [traj_ids.get_loc(s).start + 1 for s in incomplete_start]

        start_idx = start_idx + starts_
        end_idx = end_idx + ends_

        trajs['stop_move_label'] = np.nan
        
        i = 1
        for s,e in zip(start_idx,end_idx):
            trajs['stop_move_label'][s:e+1] = i
            i += 1

        trajs['stop_move_label'].ffill(inplace=True)
        trajs['stop_move_label'].fillna(0,inplace=True)

        trajs['stop_move_label'][(trajs['start_stop']==1)|(trajs['end_stop']==1)] = -1

        moves = trajs[trajs['stop_move_label']!=-1]
        self.moves = moves
        moves.to_parquet('data/temp_dataset/moves.parquet')

        del end_df, start_df, traj_df

    def get_users(self):
        
        return self.moves['uid'].unique()

    def get_trajectories(self, uid):

        return len(self.moves[self.moves['uid']==uid]['tid'].unique())

class stop_move_enrichment(Enrichment):

    '''
    `stop_move_enrichment` is a subclass of `Enrichment` module. This subclass allows users to:
    1) enrich moves with transportation mean
    2) enrich stops labeling them as occasional and systematic ones
        2.a) occasional stops are enriched with PoIs, weather, etc.
        2.b) systematic stops are enriched as home/work or other
    '''

    pass

    moves = pd.DataFrame()

    def __init__(self):
        
        self.moves = pd.read_parquet('data/temp_dataset/moves.parquet')

    def moves_enrichment(self):

        # load random forest classifier
        model = pickle.load(open('models/best_rf.sav', 'rb'))

        # preparing input for transport mode detection
        trajs = self.moves.copy()
        trajs.fillna(0,inplace=True)
        trajs.set_index(['traj_id','label'],inplace=True)

