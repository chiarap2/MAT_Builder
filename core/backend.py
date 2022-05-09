from multiprocessing.dummy import active_children
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
from sklearn.preprocessing import StandardScaler
import geohash2



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

        trajs['move_id'] = np.nan
        
        i = 1
        for s,e in zip(start_idx,end_idx):
            trajs['move_id'][s:e+1] = i
            i += 1

        trajs['move_id'].ffill(inplace=True)
        trajs['move_id'].fillna(0,inplace=True)

        trajs['move_id'][(trajs['start_stop']==1)|(trajs['end_stop']==1)] = -1

        moves = trajs[trajs['move_id']!=-1]
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

    stops = pd.DataFrame()
    moves = pd.DataFrame()

    def __init__(self,list_):
        
        self.moves = pd.read_parquet('data/temp_dataset/moves.parquet')

        if list_[6] == ['yes']:
            self.enrich_moves = True
        else:
            self.enrich_moves = False

    global df
    def core(self):

        #################################
        ### ---- MOVE ENRICHMENT ---- ###
        #################################

        if self.enrich_moves == True:
            
            moves = self.moves

            # add speed, acceleration, distance info using PTRAIL
            df = PTRAILDataFrame(moves, latitude='lat', longitude='lng', datetime='datetime', traj_id='tid')
            speed = spatial.create_speed_column(df)
            bearing = spatial.create_bearing_rate_column(speed)
            acceleration = spatial.create_jerk_column(bearing)

            # save acceleration
        
            # load random forest classifier
            model = pickle.load(open('models/best_rf.sav', 'rb'))

            # preparing input for transport mode detection
            acceleration.reset_index(inplace=True)
            acceleration.set_index(['traj_id','move_id'],inplace=True)

            acceleration2 = acceleration.copy()

            acceleration['tot_distance'] = acceleration.groupby(['traj_id','move_id']).apply(lambda x: x['Distance'].sum())
            acceleration = acceleration.groupby(['traj_id','move_id']).mean()
            acceleration[['max_bearing','max_bearing_rate','max_speed','max_acceleration','max_jerk']] = acceleration2.groupby(['traj_id','move_id']).apply(lambda x: x[['Bearing', 'Bearing_Rate', 'Speed', 'Acceleration', 'Jerk']].max())

            col_to_train = ['Bearing', 'Bearing_Rate', 'Speed', 'Acceleration', 'Jerk','tot_distance','max_bearing','max_bearing_rate','max_speed','max_acceleration','max_jerk']
            col = acceleration.columns.to_list()
            col_to_drop = [c for c in col if c not in col_to_train]
            acceleration.drop(columns=col_to_drop, inplace=True)
            acceleration.fillna(0,inplace=True)

            scaler = StandardScaler()
            scaled = scaler.fit_transform(acceleration)

            transport_pred = model.predict(scaled)
            acceleration['label'] = transport_pred

            # transport label:
            # 0: walk
            # 1: bike
            # 2: bus
            # 3: car
            # 4: subway
            # 5: train
            # 6: taxi

            moves.set_index(['tid','move_id'],inplace=True)
            moves_index = moves.index
            acceleration_index = acceleration.index
            moves.loc[moves_index.isin(acceleration_index),'label'] = acceleration.loc[acceleration_index.isin(moves_index),'label']

        
        ############################################
        ### ---- SYSTEMATIC STOP ENRICHMENT ---- ###
        ############################################

        self.stops = pd.read_parquet('data/temp_dataset/stops.parquet')

        stops = self.stops.copy()
        stops['pos_hashed'] = stops[['lat','lng']].apply(lambda x: geohash2.encode(x[0],x[1],7),raw=True,axis=1)
        stops['frequency'] = 0

        def compute_freq(x):
            freqs = x.groupby('pos_hashed').count()['frequency']
            return freqs

        systematic_sp = pd.DataFrame(stops.groupby('uid').apply(lambda x: compute_freq(x)))
        sp = stops.copy()
        sp.set_index(['uid','pos_hashed'],inplace=True)
        sp['frequency'] = systematic_sp['frequency']
        sp.reset_index(inplace=True)
        systematic_stops = sp[sp['frequency']>2]
        #print(systematic_stops)
        
        systematic_stops['start_time'] = systematic_stops['datetime'].dt.hour
        systematic_stops['end_time'] = systematic_stops['leaving_datetime'].dt.hour

        #global freq
        freq = pd.DataFrame(np.zeros((len(systematic_stops),24)))
        freq['uid'] = systematic_stops['uid']
        freq['location'] = systematic_stops['pos_hashed']
        freq.drop_duplicates(['uid','location'],inplace=True)
                
        def update_hour(x):
            
            start_col = x[-2]
            end_col = x[-1]
            
            start_raw = freq[(freq['uid']==x[0])&(freq['location']==x[1])].first_valid_index()
            if start_raw != None:
                end_raw = start_raw+1
                
                if start_col<end_col:
                    
                    freq.loc[start_raw:end_raw,start_col:end_col] += 1
                
                elif start_col==end_col:
                    
                    freq.loc[start_raw:end_raw,start_col] += 1
                    
                else:
                    
                    freq.loc[start_raw:end_raw,start_col:23] += 1
                    freq.loc[start_raw:end_raw,0:end_col] += 1

        systematic_stops.apply(lambda x: update_hour(x),raw=True,axis=1)
        hours = [i for i in range(0,24)]
        freq['sum'] = freq[hours].sum(axis=1)
        freq['tot'] = freq.groupby('uid')['sum'].sum()
        freq['importance'] = freq['sum'] / freq['tot']

        freq.set_index(['uid','location'],inplace=True)
        freq.drop(columns=['sum','tot'],inplace=True)

        freq['night'] = freq[[23,0,1,2,3,4,5]].sum(axis=1)
        freq['morning'] = freq[[6,7,8,9,10,11,12]].sum(axis=1)
        freq['afternoon'] = freq[[13,14,15,16,17,18]].sum(axis=1)
        freq['evening'] = freq[[19,20,21,22]].sum(axis=1)

        largest = pd.DataFrame(freq.groupby('uid')['importance'].nlargest(2))
        largest.index = largest.index.droplevel(0)
        freq['home'] = 0
        freq['work'] = 0
        freq['other'] = 0

        w_home = [0.6,0.1,0.1,0.4]
        w_work = [0.1,0.6,0.4,0.1]

        freq['p_home'] = (freq[['night','morning','afternoon','evening']].loc[largest.index] * w_home).sum(axis=1)
        freq['p_work'] = (freq[['night','morning','afternoon','evening']].loc[largest.index] * w_work).sum(axis=1)
        freq['home'] = freq['p_home'] / freq[['p_home','p_work']].sum(axis=1)
        freq['work'] = freq['p_work'] / freq[['p_home','p_work']].sum(axis=1)