import geopandas as gpd
import pandas as pd
import numpy as np

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.kinematic_features import KinematicFeatures as spatial

from sklearn.cluster import DBSCAN

import pickle
from sklearn.preprocessing import StandardScaler
import geohash2
import osmnx as ox

from core.ModuleInterface import ModuleInterface
from core.RDF_builder import RDFBuilder


class Enrichment(ModuleInterface):
    '''
    `Enrichment` is a class that models the semantic enrichment module. An instance of this class:

    1) enriches the trajectories with the moves, along with the estimated transportation means
    2) enriches the trajectories with stops, either occasional or systematic. Moreover:
        2.a) occasional stops are further augmented with PoIs, weather, etc.
        2.b) systematic stops are further augmented with labels indicating whether they represent home, work, or other.
    3) enriches the trajectories with weather information.
    4) enriches the trajectory users with the social media posts they have written.

    Finally, the class uses the RDF_Builder class to build a knowledge graph containing the enriched trajectories.
    The KG is populated according to a customized version of the STEPv2 ontology.
    '''

    ### CLASS PUBLIC STATIC FIELDS ###

    id_class = 'Enrichment'



    ### CLASS PROTECTED METHODS ###

    ### METHODS RELATED TO THE MOVES ENRICHMENT ###
    def _moves_enrichment(self, moves, model) :
        # add speed, acceleration, distance info using PTRAIL
        df = PTRAILDataFrame(moves, latitude='lat',
                             longitude='lng',
                             datetime='datetime',
                             traj_id='tid')

        speed = spatial.create_speed_column(df)
        bearing = spatial.create_bearing_rate_column(speed)
        acceleration = spatial.create_jerk_column(bearing)

        # preparing input for transport mode detection
        acceleration.reset_index(inplace=True)
        acceleration.set_index(['traj_id', 'move_id'], inplace=True)

        acceleration2 = acceleration.copy()

        acceleration['tot_distance'] = acceleration.groupby(['traj_id', 'move_id']).apply(lambda x: x['Distance'].sum())
        acceleration = acceleration.groupby(['traj_id', 'move_id']).mean()
        acceleration[
            ['max_bearing', 'max_bearing_rate', 'max_speed', 'max_acceleration', 'max_jerk']] = acceleration2.groupby(
            ['traj_id', 'move_id']).apply(lambda x: x[['Bearing', 'Bearing_Rate', 'Speed', 'Acceleration', 'Jerk']].max())

        col_to_train = ['Bearing', 'Bearing_Rate', 'Speed', 'Acceleration', 'Jerk', 'tot_distance', 'max_bearing',
                        'max_bearing_rate', 'max_speed', 'max_acceleration', 'max_jerk']
        col = acceleration.columns.to_list()
        col_to_drop = [c for c in col if c not in col_to_train]
        acceleration.drop(columns=col_to_drop, inplace=True)
        acceleration.fillna(0, inplace=True)

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

        moves.set_index(['tid', 'move_id'], inplace=True)
        moves_index = moves.index
        acceleration_index = acceleration.index
        moves.loc[moves_index.isin(acceleration_index), 'label'] = \
            acceleration.loc[acceleration_index.isin(moves_index), 'label']

        return moves

    ### METHODS RELATED TO THE OCCASIONAL/SYSTEMATIC STOPS ENRICHMENT ###

    def _download_poi_osm(self, list_pois: list[str], place: str, write_files: bool = False) -> gpd.GeoDataFrame:

        # Final list of the columns that are expected to be found in the POI dataframe.
        list_columns_df_poi = ['osmid', 'element_type', 'name', 'name:en', 'wikidata', 'geometry', 'category']

        # Here we download the POIs from OSM if the list of types of POIs is not empty.
        gdf_ = gpd.GeoDataFrame(columns=list_columns_df_poi, crs="EPSG:4326")
        if list_pois:

            print(f"Downloading POIs from OSM for the location {place}...")
            for key in list_pois:

                # downloading POI
                print(f"Downloading {key} POIs from OSM...")
                poi = ox.geometries_from_place(place, tags={key: True})
                print(f"Download completed!")

                # Immediately return the empty dataframe if it doesn't contain any suitable POI...
                if poi.empty:
                    print(f"No POI found for category {key}!")
                    break

                # Remove the POIs that do not have a name.
                poi.reset_index(inplace=True)
                poi.rename(columns={key: 'category'}, inplace=True)
                poi.drop(poi.columns.difference(list_columns_df_poi), axis=1, inplace=True)
                poi = poi.loc[~poi['name'].isna()]
                poi['category'].replace({'yes': key}, inplace=True)

                # Now write out this subset of POIs to a file.
                if write_files: poi.to_parquet('./' + key + '.parquet')

                # And finally, concatenate this subset of POIs to the other POIs
                # that have been added to the main dataframe so far.
                gdf_ = pd.concat([gdf_, poi])

            gdf_.reset_index(drop=True, inplace=True)
            if write_files: gdf_.to_parquet('./pois.parquet')
            return gdf_

    ### METHODS RELATED TO THE SYSTEMATIC STOPS ENRICHMENT ###

    def _systematic_enrichment_geohash(self, stops : pd.DataFrame, geohash_precision : int, min_frequency_sys : int) -> pd.DataFrame :

        # Qui mappiamo i centroidi degli stop vs le celle materializzate tramite la funzione di geohash.
        stops['pos_hashed'] = [geohash2.encode(lat, lng, geohash_precision) for lat, lng in zip(stops['lat'], stops['lng'])]


        # Conta la frequenza delle coppie ('uid', 'pos_hashed').
        # NOTA: questa e' la base da cui individueremo i systematic stop e li classificheremo.
        systematic_sp = pd.DataFrame(stops.groupby(['uid', 'pos_hashed']).size(), columns = ['frequency']).reset_index()
        # display(systematic_sp)


        # Ora associa gli stop originali alle coppie (uid, pos_hashed) insieme alla frequenza trovata.
        sp = stops.merge(systematic_sp, on = ['uid', 'pos_hashed'], how = 'left')
        # display(sp)


        systematic_stops = sp[sp['frequency'] >= min_frequency_sys].copy()
        systematic_stops['systematic_id'] = systematic_stops.groupby(["uid", "pos_hashed"]).ngroup()
        systematic_stops.reset_index(inplace=True, drop=True)
        # display(systematic_stops)


        # 1 - case where systematic stops have been found...
        if systematic_stops.shape[0] != 0:

            # 1.1 - Prepare the dataframe that will hold the hours' frequencies.
            freq = pd.DataFrame(np.zeros((len(systematic_stops), 24), dtype=np.uint32))
            freq['weekend'] = 0
            freq['uid'] = systematic_stops['uid']
            freq['location'] = systematic_stops['pos_hashed']
            freq.drop_duplicates(['uid', 'location'], inplace=True)
            # display(freq)

            # 1.2 - Qui calcoliamo i contatori delle ore in cui occorrono i sistematic stop
            # (propedeutico a determinare se si tratta di home/work/other).
            def update_freq(freq, systematic_stops):
                it = zip(systematic_stops['uid'], systematic_stops['pos_hashed'],
                         systematic_stops['datetime'], systematic_stops['leaving_datetime'])

                for uid, location, start, end in it:
                    time_range = pd.date_range(start.floor('h'), end.floor('h'), freq='H')
                    indexer = (freq['uid'] == uid) & (freq['location'] == location)
                    for t in time_range:
                        if t.weekday() > 4:
                            freq.loc[indexer, 'weekend'] += 1
                        else:
                            freq.loc[indexer, t.hour] += 1

            update_freq(freq, systematic_stops)
            # display(freq)

            # 1.3 - Qui calcoliamo l'importanza degli stop sistematici trovati per ogni utente.
            #     I due piu' importanti vengono assunti essere casa o lavoro.
            slots = list(range(0, 24)) + ['weekend']
            freq['sum'] = freq[slots].sum(axis=1)  # Qui calcoliamo la somma delle ore per ogni systematic stop.
            freq['tot'] = freq.groupby('uid')['sum'].transform('sum')
            freq['importance'] = freq['sum'] / freq['tot']  # E qui determiniamo l'importanza di ogni systematic stop.
            freq.drop(columns=['sum', 'tot'], inplace=True)
            freq.set_index(['uid', 'location'], inplace=True)
            # display(freq)

            # 1.4 - Qui distribuiamo i contatori associate alle ore ai 4 momenti del giorno.
            freq['night'] = freq[[22, 23, 0, 1, 2, 3, 4, 5, 6, 7]].sum(axis=1)
            freq['morning'] = freq[[8, 9, 10, 11, 12]].sum(axis=1)
            freq['afternoon'] = freq[[13, 14, 15, 16, 17, 18]].sum(axis=1)
            freq['evening'] = freq[[19, 20, 21]].sum(axis=1)
            # display(freq)

            # 1.5 - Qui per ogni utente troviamo i primi 2 stop sistematici con importanza maggiore (saranno )
            largest_index = pd.DataFrame(freq.groupby('uid')['importance'].nlargest(2)).index.droplevel(0)
            # display(largest_index)

            # 1.6 - Qui calcoliamo le probabilita' associate alle tipologie di stop sistematici.
            w_home = [1, 0.1, 0, 0.9, 1]
            w_work = [0, 0.9, 1, 0.1, 0]
            list_exclusion = list(set(freq.index) - set(largest_index))
            freq['p_home'] = (freq[['night', 'morning', 'afternoon', 'evening', 'weekend']] * w_home).sum(axis=1)
            freq['p_work'] = (freq[['night', 'morning', 'afternoon', 'evening', 'weekend']] * w_work).sum(axis=1)
            freq['home'] = freq['p_home'] / (freq['p_home'] + freq['p_work'])
            freq['work'] = freq['p_work'] / (freq['p_home'] + freq['p_work'])
            freq.loc[list_exclusion, 'home'] = 0
            freq.loc[list_exclusion, 'work'] = 0
            freq['other'] = 0
            freq.loc[list_exclusion, 'other'] = 1
            freq.drop(columns=['p_home', 'p_work'], inplace=True)
            # display(freq)

            # 7 - Qui, infine, completiamo il dataframe systematic stops con le varie probabilita' calcolate.
            systematic_stops.set_index(['uid', 'pos_hashed'], inplace=True)
            systematic_stops['home'] = 0
            systematic_stops['work'] = 0
            systematic_stops['other'] = 1
            systematic_stops['importance'] = freq['importance']
            systematic_stops.loc[largest_index, 'home'] = freq.loc[largest_index, 'home']
            systematic_stops.loc[largest_index, 'work'] = freq.loc[largest_index, 'work']
            systematic_stops.loc[largest_index, 'other'] = freq.loc[largest_index, 'other']

            systematic_stops.reset_index(inplace=True)


        # 2 - case where no systematic stops have been found...adapt the empty dataframe
        #     for the code that will use it.
        else:
            new_cols = ['systematic_id', 'importance', 'home', 'work', 'other', 'start_time', 'end_time']
            systematic_stops = systematic_stops.reindex(systematic_stops.columns.union(new_cols), axis=1)


        # Print out the dataframe holding the systematic stops.
        return systematic_stops

    def _dbscan_stops(self, stops, eps, min_pts):

        # convert epsilon from km to radians
        kms_per_radian = 6371.0088
        eps = (eps / 1000) / kms_per_radian

        # set up the algorithm
        dbscan = DBSCAN(eps=eps,
                        min_samples=min_pts,
                        algorithm='ball_tree',
                        metric='haversine')

        # Fit the algorithm on the stops of individual users.
        df_stops = stops.copy()
        df_stops['systematic_id'] = -5
        gb = df_stops.groupby('uid')
        for k, df in gb:
            # print(f"Processing the stops of user {k}!")

            dbscan.fit(np.radians([x for x in zip(df['lat'], df['lng'])]))

            df_stops.loc[df.index, 'systematic_id'] = dbscan.labels_
            # print(f"Main dataframe after the update: {df_stops.loc[df_stops['uid'] == k, 'sys_id']}")

            view = df_stops[df_stops['uid'] == k]
            # print(f"Number of clusters found for this user: {view.loc[view['sys_id'] >= 0, 'sys_id'].nunique()}")
            # print(f"Number of occasional stops found for this user: {len(view.loc[view['sys_id'] == -1])}")
            # break

        # display(df_stops)
        # display(df_stops.info())
        return df_stops['systematic_id']

    def _systematic_enrichment_dbscan(self, stops: pd.DataFrame, epsilon: int, min_frequency_sys: int) -> pd.DataFrame:

        # Qui mappiamo i centroidi degli stop vs le celle materializzate tramite la funzione di geohash.
        systematic_sp = stops.copy()
        systematic_sp['systematic_id'] = self._dbscan_stops(stops, epsilon, min_frequency_sys) # Assign the stops to the clusters.
        systematic_sp = systematic_sp.loc[systematic_sp['systematic_id'] >= 0, :] # Eliminate the stops that do not belong to any cluster.
        # print(systematic_sp)

        # Conta la frequenza delle coppie ('uid', 'pos_hashed').
        # NOTA: questa e' la base da cui individueremo i systematic stop e li classificheremo.
        freq_sys_sp = pd.DataFrame(systematic_sp.groupby(['uid', 'systematic_id']).size(),
                                   columns=['frequency']).reset_index()
        # display(freq_sys_sp)

        # Ora associa gli stop originali alle coppie (uid, pos_hashed) insieme alla frequenza trovata.
        systematic_sp = systematic_sp.merge(freq_sys_sp, on=['uid', 'systematic_id'], how='left')
        # display(systematic_sp)
        # display(systematic_sp['id_sys'].unique())

        # 1 - case where systematic stops have been found...
        if systematic_sp.shape[0] != 0:

            # 1.1 - Prepare the dataframe that will hold the hours' frequencies.
            freq = pd.DataFrame(np.zeros((len(systematic_sp), 24), dtype=np.uint32))
            freq['weekend'] = 0
            freq['uid'] = systematic_sp['uid']
            freq['location'] = systematic_sp['systematic_id']
            freq.drop_duplicates(['uid', 'location'], inplace=True)
            # display(freq)


            # 1.2 - Qui calcoliamo i contatori delle ore in cui occorrono i sistematic stop
            # (propedeutico a determinare se si tratta di home/work/other).
            def update_freq(freq, systematic_sp):
                it = zip(systematic_sp['uid'], systematic_sp['systematic_id'],
                         systematic_sp['datetime'], systematic_sp['leaving_datetime'])

                for uid, location, start, end in it:
                    time_range = pd.date_range(start.floor('h'), end.floor('h'), freq='H')
                    indexer = (freq['uid'] == uid) & (freq['location'] == location)
                    for t in time_range:
                        if t.weekday() > 4:
                            freq.loc[indexer, 'weekend'] += 1
                        else:
                            freq.loc[indexer, t.hour] += 1

            update_freq(freq, systematic_sp)
            # display(freq)


            # 1.3 - Qui calcoliamo l'importanza degli stop sistematici trovati per ogni utente.
            #     I due piu' importanti vengono assunti essere casa o lavoro.
            slots = list(range(0, 24)) + ['weekend']
            freq['sum'] = freq[slots].sum(axis=1)  # Qui calcoliamo la somma delle ore per ogni systematic stop.
            freq['tot'] = freq.groupby('uid')['sum'].transform('sum')
            freq['importance'] = freq['sum'] / freq['tot']  # E qui determiniamo l'importanza di ogni systematic stop.
            freq.drop(columns=['sum', 'tot'], inplace=True)
            freq.set_index(['uid', 'location'], inplace=True)
            # display(freq)


            # 1.4 - Qui distribuiamo i contatori associate alle ore ai 4 momenti del giorno.
            freq['night'] = freq[[22, 23, 0, 1, 2, 3, 4, 5, 6, 7]].sum(axis=1)
            freq['morning'] = freq[[8, 9, 10, 11, 12]].sum(axis=1)
            freq['afternoon'] = freq[[13, 14, 15, 16, 17, 18]].sum(axis=1)
            freq['evening'] = freq[[19, 20, 21]].sum(axis=1)
            # display(freq)


            # 1.5 - Qui per ogni utente troviamo i primi 2 stop sistematici con importanza maggiore (saranno )
            largest_index = pd.DataFrame(freq.groupby('uid')['importance'].nlargest(2)).index.droplevel(0)
            # display(largest_index)


            # 1.6 - Qui calcoliamo le probabilita' associate alle tipologie di stop sistematici.
            w_home = [1, 0.1, 0, 0.9, 1]
            w_work = [0, 0.9, 1, 0.1, 0]
            list_exclusion = list(set(freq.index) - set(largest_index))
            freq['p_home'] = (freq[['night', 'morning', 'afternoon', 'evening', 'weekend']] * w_home).sum(axis=1)
            freq['p_work'] = (freq[['night', 'morning', 'afternoon', 'evening', 'weekend']] * w_work).sum(axis=1)
            freq['home'] = freq['p_home'] / (freq['p_home'] + freq['p_work'])
            freq['work'] = freq['p_work'] / (freq['p_home'] + freq['p_work'])
            freq.loc[list_exclusion, 'home'] = 0
            freq.loc[list_exclusion, 'work'] = 0
            freq['other'] = 0
            freq.loc[list_exclusion, 'other'] = 1
            freq.drop(columns=['p_home', 'p_work'], inplace=True)
            # display(freq)


            # 1.7 - Qui, infine, completiamo il dataframe systematic stops con le varie probabilita' calcolate.
            systematic_sp.set_index(['uid', 'systematic_id'], inplace=True)
            systematic_sp['home'] = 0
            systematic_sp['work'] = 0
            systematic_sp['other'] = 1
            systematic_sp['importance'] = freq['importance']
            systematic_sp.loc[largest_index, 'home'] = freq.loc[largest_index, 'home']
            systematic_sp.loc[largest_index, 'work'] = freq.loc[largest_index, 'work']
            systematic_sp.loc[largest_index, 'other'] = freq.loc[largest_index, 'other']

            systematic_sp.reset_index(inplace=True)


        # 2 - case where no systematic stops have been found...adapt the empty dataframe
        #     for the code that will use it.
        else:
            new_cols = ['systematic_id', 'importance', 'home', 'work', 'other', 'start_time', 'end_time']
            systematic_sp = systematic_sp.reindex(systematic_sp.columns.union(new_cols), axis=1)

        # Print out the dataframe holding the systematic stops.
        return systematic_sp

    ### METHODS RELATED TO THE OCCASIONAL STOPS ENRICHMENT ###

    def _stop_enrichment_with_pois(self,
                                   df_stops : pd.DataFrame,
                                   df_poi : gpd.GeoDataFrame,
                                   suffix : str, max_distance : float) -> gpd.GeoDataFrame:

        # print("DEBUG enrichment stops...")

        # Prepare the stops for the subsequent spatial join.
        stops = gpd.GeoDataFrame(df_stops,
                                 geometry=gpd.points_from_xy(df_stops.lng, df_stops.lat),
                                 crs="EPSG:4326")
        stops.to_crs('epsg:3857', inplace=True)
        stops['geometry_stop'] = stops['geometry']
        stops['geometry'] = stops['geometry_stop'].buffer(max_distance)


        pois = df_poi.copy()
        pois.to_crs('epsg:3857', inplace=True)

        # Filter out the POIs without a name!
        pois = pois.loc[pois['name'].notna(), :]

        # duplicate geometry column because we loose it during the sjoin_nearest
        pois['geometry_' + suffix] = pois['geometry']
        pois['element_type'] = pois['element_type'].astype(str)
        pois['osmid'] = pois['osmid'].astype(str)

        # print(f"Stampa df stop occasionali: {stop.info()}")
        print(f"Stampa df POIs: {pois.info()}")

        # Execute the spatial left join to associate POIs to the stops.
        enriched_stops = stops.sjoin_nearest(pois, max_distance=0.00001, how='left', rsuffix=suffix)

        # Remove the POIs that have been associated with the same stop multiple times.
        enriched_stops.drop_duplicates(subset=['stop_id', 'osmid'], inplace=True)

        # compute the distance between the stop point and the POI geometry
        enriched_stops['distance'] = enriched_stops['geometry_stop'].distance(enriched_stops['geometry_' + suffix])

        # NOTE: Keep the rows for which it was not possible to associate a stop to a POI.

        # Sort by distance
        enriched_stops = enriched_stops.sort_values(['tid', 'stop_id', 'distance'])
        enriched_stops.reset_index(drop = True, inplace = True)

        # print(f"Stampa df risultati: {enriched_stops.info()}")
        return enriched_stops



    ### CLASS PUBLIC CONSTRUCTOR ###
    
    def __init__(self) :

        # Here we initialize the various class fields.
        print(f"Executing constructor of class {self.id_class}!")
        self.reset_state()
        
    
    
    ### CLASS PUBLIC METHODS ###
        
    def execute(self, dic_params : dict) -> bool :

        # Parsing the input received from the UI / user...

        # 0 - Trajectories
        self.trajectories = pd.DataFrame(dic_params['trajectories'])

        # 1 - moves parameters
        self.moves = dic_params['moves']
        if dic_params['move_enrichment']:
            self.enrich_moves = True
        else:
            self.enrich_moves = False


        # 2 - Stops and POIs parameters
        self.stops = dic_params['stops']
        self.poi_place = dic_params['poi_place']
        self.list_pois = [] if dic_params['poi_categories'] is None else dic_params['poi_categories']
        self.path_poi = dic_params['path_poi']
        self.max_distance = dic_params['max_dist']
        self.dbscan_epsilon = dic_params['dbscan_epsilon']
        self.systematic_threshold = dic_params['systematic_threshold']

        # 3 - Social media posts parameters
        if dic_params['social_enrichment'] is None:
            self.tweet_user = False
        else:
            self.tweet_user = True
            self.upload_social = dic_params['social_enrichment']

        # 4 - weather parameters
        if dic_params['weather_enrichment'] is None:
            self.weather = False
        else:
            self.weather = True
            self.upload_weather = dic_params['weather_enrichment']

        # 5 - RDF knowledge graph creation parameters
        self.rdf = dic_params['create_rdf']


        # 6 - Core execution.
        return self.core()


    def core(self) -> bool :
    
        print("Executing the core of the semantic enrichment module...")


        #################################
        ### ---- MOVE ENRICHMENT ---- ###
        #################################

        # 1 - Case in which we augment the moves with the estimated transportation means.
        if self.enrich_moves:
            print("Executing move enrichment...")

            # load random forest classifier
            model = pickle.load(open('models/best_rf.sav', 'rb'))
            self.moves = self._moves_enrichment(self.moves.copy(), model)
            self.moves.to_parquet('data/enriched_moves.parquet')

        # 2 - Case in which we do not augment the moves: just make the dataframe compatible with the subsequent steps.
        else :
            self.moves.set_index(['tid', 'move_id'], inplace=True)
            self.moves.to_parquet('data/enriched_moves.parquet')



        ##################################################
        ###        ---- STOP ENRICHMENT PHASE ----     ###
        ##################################################

        # Here stops' index contains the IDs of the stops. We reset the index such
        # that the old index becomes a column.
        self.stops.reset_index(inplace=True)
        self.stops.rename(columns={'index': 'stop_id'}, inplace=True)

        # Get a POI dataset, either from OSM or from a file.
        df_poi = None
        if self.path_poi is None:
            print(
                f"Downloading POIs from OSM for the location of {self.poi_place}. Selected types of POIs: {self.list_pois}")
            df_poi = self._download_poi_osm(self.list_pois, self.poi_place)
        else:
            df_poi = self.path_poi
            print(f"Using a POI file: {df_poi}!")
        print(f"A few info on the POIs that will be used to enrich the occasional stops: {df_poi.info()}")


        ############################################
        ### ---- SYSTEMATIC STOP ENRICHMENT ---- ###
        ############################################
        
        print("Executing systematic stop detection...")
        #self.systematic = self._systematic_enrichment_geohash(self.stops.copy(),
        #                                                      geohash_precision = self.geohash_precision,
        #                                                      min_frequency_sys = self.systematic_threshold)
        self.systematic = self._systematic_enrichment_dbscan(self.stops.copy(),
                                                             epsilon = self.dbscan_epsilon,
                                                             min_frequency_sys = self.systematic_threshold)

        print("Executing systematic stop augmentation with POIs...")
        self.enriched_systematic = self._stop_enrichment_with_pois(self.systematic,
                                                                   df_poi,
                                                                   'poi',
                                                                   self.max_distance)
        self.enriched_systematic.to_parquet('data/enriched_systematic.parquet')


        ############################################
        ### ---- OCCASIONAL STOP ENRICHMENT ---- ###
        ############################################

        print("Executing occasional stop augmentation with POIs...")
        self.occasional = self.stops[~self.stops['stop_id'].isin(self.systematic['stop_id'])]

        # Calling functions internal to this method...
        self.enriched_occasional = self._stop_enrichment_with_pois(self.occasional,
                                                                   df_poi,
                                                                   'poi',
                                                                   self.max_distance)
        # mat.set_index(['stop_id','lat','lng'],inplace=True)
        self.enriched_occasional.to_parquet('data/enriched_occasional.parquet')
        
        
        
        ####################################
        ### ---- WEATHER ENRICHMENT ---- ###
        ####################################       

        def move_weather_enrichment(moves, weather) :

            res = moves.copy()
            if weather is not None :
                
                res['DATE'] = (res['datetime'].dt.date).astype(str)
                res = res.merge(weather, on = 'DATE', how = 'left')
                res.drop(columns = ['DATE'])
                res.rename(columns = {'TAVG_C' : 'temperature', 'DESCRIPTION' : 'w_conditions'}, inplace = True)
                
                # The merge has eliminated the old multilevel index on moves...let's restore it. 
                res = res.set_index(moves.index) 

            # If weather conditions are not available then set NA values in the appropriate columns.
            else :
                res['temperature'] = pd.NA
                res['w_conditions'] = pd.NA

            return(res)
        
        
        def weather_enrichment(traj_cleaned, weather) :
            
            traj_cleaned['DATE'] = (traj_cleaned['datetime'].dt.date).astype(str)
            
            # Per ogni traiettoria, trova il primo ed ultimo sample in ogni giornata coperta dalla traiettoria.
            gb = traj_cleaned.groupby(['tid', 'DATE'])
            traj_day = gb.first().reset_index()
            end_traj_day = gb.last().reset_index()

            # Adesso joina i due dataframe.
            traj_day['end_lat'] = end_traj_day['lat']
            traj_day['end_lng'] = end_traj_day['lng']
            traj_day['end_datetime'] = end_traj_day['datetime']
            print(traj_day)
            print(traj_day.info())
            
            # For each trajectory and day covered by the trajectory, find whether we can associate weather information.
            weather_enrichment = traj_day.merge(weather, on = 'DATE', how = 'inner')

            weather_enrichment['datetime'] = pd.to_datetime(weather_enrichment['datetime'], utc = True)
            weather_enrichment['end_datetime'] = pd.to_datetime(weather_enrichment['end_datetime'], utc = True)

            print(weather_enrichment)
            print(weather_enrichment.info())

            return weather_enrichment
        
            
            
        df_weather_enrichment = None
        print(f"Valore self.weather: {self.weather}")
        if self.weather :
            print("Adding weather info to the trajectories...")

            weather = self.upload_weather
            
            df_weather_enrichment = weather_enrichment(self.trajectories.copy(), weather)
            df_weather_enrichment.to_parquet('./data/weather_enrichment.parquet')
        
            # Set up the moves dataframe to have temperature and weather conditions (needed later on by
            # the component plotting the trajectories).
            self.moves = move_weather_enrichment(self.moves, weather)
        else :
            self.moves = move_weather_enrichment(self.moves, None)
        
        
        
        ##############################################
        ### ---- SOCIAL MEDIA POST ENRICHMENT ---- ###
        ##############################################
        
        tweets_RDF = None
        print(f"Valore self.tweet_user: {self.tweet_user}")        
        if self.tweet_user :
            print("Enriching users with social media posts!")
            
            tweets = self.upload_social
            tweets_RDF = tweets.copy()
            
            self.moves['date'] = self.moves['datetime'].dt.date
            tweets['tweet_created'] = tweets['tweet_created'].astype('datetime64')
            tweets['tweet_created'] = tweets['tweet_created'].dt.date
            self.moves['tweet'] = ''

            self.moves.reset_index(inplace=True)  

            self.moves.set_index(['date','uid'],inplace=True)   
            tweets.set_index(['tweet_created','uid'],inplace=True)      
            matched_tweets = self.moves.join(tweets,how='inner')

            self.moves.reset_index(inplace=True)
            matched_tweets.reset_index(inplace=True)

            self.tweets = matched_tweets.copy()    
        
        else :
            self.tweets = None



        # This final index reset ensures that the dataframe containing the moves will NOT have
        # its index on the trajectory identifiers. This is required as previous enrichment steps
        # modified this dataframe's index.
        # TODO: review and simplify the code to eliminate this need.
        self.moves.reset_index(inplace=True)



        ##############################################
        ###       RDF KNOWLEDGE GRAPH SAVE         ###
        ##############################################
        
        if self.rdf :
            
            print("Creating and then saving to disk the RDF graph...")

            # Instantiate RDF-builder
            builder = RDFBuilder()

            # Add the users and the information associated to their raw-trajectories to the graph.
            builder.add_trajectories(self.trajectories.copy())

            # Add to the RDF graph the stops, the moves, and the semantic information 
            # associated with the trajectories (social media posts, weather), the stops (type of stop, POIs),
            # and the moves (transportation mean).
            builder.add_moves(self.moves, self.enrich_moves)
            builder.add_occasional_stops(self.enriched_occasional)
            builder.add_systematic_stops(self.enriched_systematic)
            
            # Add weather information to the trajectories.
            if df_weather_enrichment is not None :
                builder.add_weather(df_weather_enrichment)
                
            # Add weather information to the trajectories.
            if tweets_RDF is not None :
                builder.add_social(tweets_RDF)
            
            # Output the RDF graph to disk in Turtle format
            print("Saving the KG to disk!")
            builder.serialize_graph('kg.ttl')

        print("Enrichment complete!")
           
    def get_results(self) -> dict:
        return {'trajectories' : self.trajectories if self.trajectories is not None else None,
                'moves' : self.moves.copy() if self.moves is not None else None,
                'occasional' : self.occasional.copy() if self.occasional is not None else None,
                'systematic' : self.systematic.copy() if self.systematic is not None else None,
                'enriched_systematic': self.enriched_systematic.copy() if self.enriched_systematic is not None else None,
                'enriched_occasional' : self.enriched_occasional.copy() if self.enriched_occasional is not None else None,
                'tweets' : self.tweets.copy() if self.tweets is not None else None}

    def get_params_input(self) -> list[str] :
        return ['trajectories',
                'move_enrichment',
                'place',
                'poi_categories',
                'path_poi',
                'max_dist',
                'dbscan_epsilon',
                'systematic_threshold',
                'social_enrichment',
                'weather_enrichment',
                'create_rdf']

    def get_params_output(self) -> list[str] :
        return list(self.get_results().keys())
        
    def reset_state(self) :
        # These are the auxiliary fields internally used during the enrichment execution.
        self.dbscan_epsilon = None
        self.systematic_threshold = None
        self.rdf = None
        self.upload_weather = None
        self.weather = None
        self.upload_social = None
        self.tweet_user = None
        self.max_distance = None
        self.path_poi = None
        self.list_pois = None
        self.poi_place = None
        self.enrich_moves = None

        # These are the fundamental fields internally used during the enrichment execution, and that may
        # be exposed by a class instance (e.g., if the instance is used via a UI wrapper).
        self.trajectories = None
        self.stops = None
        self.moves = None

        self.enriched_systematic = None
        self.enriched_occasional = None
        self.systematic = None
        self.occasional = None
        self.tweets = None
