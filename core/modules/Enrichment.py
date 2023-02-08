import geopandas as gpd
import pandas as pd
import numpy as np

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.kinematic_features import KinematicFeatures as spatial

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

    ### CLASS PROTECTED METHODS ###

    def _systematic_enrichment(self, stops : pd.DataFrame, geohash_precision : int, min_frequency_sys : int) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Qui l'indice in 'stops' e' l'id degli stop...resettiamo l'indice in maniera tale da farlo diventare
        # la colonna degli ID degli stop.
        stops.reset_index(inplace=True)
        stops.rename(columns={'index': 'stop_id'}, inplace=True)


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
            systematic_stops['start_time'] = systematic_stops['datetime'].dt.hour
            systematic_stops['end_time'] = systematic_stops['leaving_datetime'].dt.hour


            # 1.1 - Prepare the dataframe that will hold the hours' frequencies.
            freq = pd.DataFrame(np.zeros((len(systematic_stops), 24), dtype = np.uint32))
            freq['uid'] = systematic_stops['uid']
            freq['location'] = systematic_stops['pos_hashed']
            freq.drop_duplicates(['uid', 'location'], inplace=True)
            # display(freq)


            # 1.2 - Qui calcoliamo i contatori delle ore in cui occorrono i sistematic stop
            # (propedeutico a determinare se si tratta di home/work/other).
            def update_freq(freq, systematic_stops) :
                it = zip(systematic_stops['uid'], systematic_stops['pos_hashed'],
                         systematic_stops['start_time'], systematic_stops['end_time'])

                for uid, location, start, end in it :
                    cols = list(range(start, end + 1)) if start <= end else list(range(0, end + 1)) + list(range(start, 24))
                    freq.loc[(freq['uid'] == uid) & (freq['location'] == location), cols] += 1

            update_freq(freq, systematic_stops)
            # display(freq)


            # 1.3 - Qui calcoliamo l'importanza degli stop sistematici trovati per ogni utente.
            #     I due piu' importanti vengono assunti essere casa o lavoro.
            hours = list(range(0, 24))
            freq['sum'] = freq[hours].sum(axis=1) # Qui calcoliamo la somma delle ore per ogni systematic stop.
            freq['tot'] = freq.groupby('uid')['sum'].transform('sum')
            freq['importance'] = freq['sum'] / freq['tot'] # E qui determiniamo l'importanza di ogni systematic stop.
            freq.drop(columns=['sum', 'tot'], inplace=True)
            freq.set_index(['uid', 'location'], inplace=True)
            # display(freq)


            # 1.4 - Qui distribuiamo i contatori associate alle ore ai 4 momenti del giorno.
            freq['night'] = freq[[22, 23, 0, 1, 2, 3, 4, 5, 6]].sum(axis=1)
            freq['morning'] = freq[[7, 8, 9, 10, 11, 12]].sum(axis=1)
            freq['afternoon'] = freq[[13, 14, 15, 16, 17, 18]].sum(axis=1)
            freq['evening'] = freq[[19, 20, 21]].sum(axis=1)
            # display(freq)


            # 1.5 - Qui per ogni utente troviamo i primi 2 stop sistematici con importanza maggiore (saranno )
            largest_index = pd.DataFrame(freq.groupby('uid')['importance'].nlargest(2)).index.droplevel(0)
            # display(largest_index)


            # 1.6 - Qui calcoliamo le probabilita' associate alle tipologie di stop sistematici.
            w_home = [0.6, 0.1, 0.1, 0.4]
            w_work = [0.1, 0.6, 0.4, 0.1]
            list_exclusion = list(set(freq.index) - set(largest_index))
            freq['p_home'] = (freq[['night', 'morning', 'afternoon', 'evening']] * w_home).sum(axis=1)
            freq['p_work'] = (freq[['night', 'morning', 'afternoon', 'evening']] * w_work).sum(axis=1)
            freq['home'] = freq['p_home'] / (freq['p_home'] + freq['p_work'])
            freq['work'] = freq['p_work'] / (freq['p_home'] + freq['p_work'])
            freq.loc[list_exclusion, 'home'] = 0
            freq.loc[list_exclusion, 'work'] = 0
            freq['other'] = 0
            freq.loc[list_exclusion, 'other'] = 1
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
        return stops, systematic_stops



    ### CLASS PUBLIC CONSTRUCTOR ###
    
    def __init__(self) :

        # These are the auxiliary fields internally used during the enrichment execution.
        self.rdf = None
        self.upload_weather = None
        self.weather = None
        self.upload_social = None
        self.tweet_user = None
        self.max_distance = None
        self.semantic_granularity = 80 # NOTE: this is a fixed, internal parameter!
        self.path_poi = None
        self.list_pois = None
        self.poi_place = None
        self.enrich_moves = None

        # These are the fundamental fields internally used during the enrichment execution, and that may
        # be exposed by a class instance (e.g., it the instance is used via a UI wrapper).
        self.stops = None
        self.moves = None

        self.mats = None
        self.systematic = None
        self.occasional = None
        self.tweets = None
        
    
    
    ### CLASS PUBLIC METHODS ###
        
    def execute(self, dic_params : dict) -> bool :

        # Parsing the input received from the UI / user...

        # 1 - moves parameters
        if dic_params['move_enrichment']:
            self.enrich_moves = True
            self.moves = dic_params['moves']
        else:
            self.enrich_moves = False


        # 2 - Stops and POIs parameters
        self.stops = dic_params['stops']
        self.poi_place = dic_params['poi_place']
        self.list_pois = [] if dic_params['poi_categories'] is None else dic_params['poi_categories']
        self.path_poi = dic_params['path_poi']
        self.max_distance = dic_params['max_dist']

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

        if self.enrich_moves:
            print("Executing move enrichment...")
            
            # add speed, acceleration, distance info using PTRAIL
            df = PTRAILDataFrame(self.moves, latitude='lat',
                                 longitude='lng',
                                 datetime='datetime',
                                 traj_id='tid')
            speed = spatial.create_speed_column(df)
            bearing = spatial.create_bearing_rate_column(speed)
            acceleration = spatial.create_jerk_column(bearing)

            ### save acceleration ###
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

            self.moves.set_index(['tid', 'move_id'], inplace=True)
            moves_index = self.moves.index
            acceleration_index = acceleration.index
            self.moves.loc[moves_index.isin(acceleration_index), 'label'] =\
                acceleration.loc[acceleration_index.isin(moves_index), 'label']

            self.moves.to_parquet('data/enriched_moves.parquet')



        ############################################
        ###        ---- STOP ENRICHMENT ----     ###
        ############################################

        ############################################
        ### ---- SYSTEMATIC STOP ENRICHMENT ---- ###
        ############################################
        
        print("Executing systematic stop enrichment...")
        # TODO: passare la soglia oltre la quale si individua uno stop sistematico.
        stops, self.systematic = self._systematic_enrichment(self.stops.copy(), 7, 5)

        
        
        ############################################
        ### ---- OCCASIONAL STOP ENRICHMENT ---- ###
        ############################################

        def select_columns(gdf, threshold=80.0) :
            """
            A function to select columns of a GeoDataFrame that have a percentage of null values
            lower than a given threshold.
            Returns a GeoDataFrame

            -----------------------------
            gdf: a GeoDataFrame

            threshold: default 80.0
                a float representing the maxium percentage of null values in a column.
            """
            
            # list of columns
            cols = gdf.columns
            # print(f"Initial set of POI columns...{cols}")
            
            # list of columns to delete
            del_cols = []
            for c in cols:

                # check if column contains only null value
                if(gdf[c].isnull().all()):
                    # save the empty column
                    del_cols.append(c)
                    continue

                # control only columns with at least one null value
                if(gdf[c].isnull().any()):

                    # compute number of null values
                    null_vals = gdf[c].isnull().value_counts()[1]
                    # compute percentage w.r.t. total number of sample
                    perc = null_vals/len(gdf[c])*100

                    # save only columns that have a perc of null values lower than the given threshold
                    if(threshold <= perc):
                        del_cols.append(c)

        
            # Now, drop the selected columns MINUS 'osmid' and 'wikidata' 
            del_cols = list(set(del_cols) - set(['osmid', 'wikidata']))
            gdf = gdf.drop(columns = del_cols)

            # print(f"POI dataframe info: {gdf.info()}")
            return gdf



        def preparing_stops(stop,max_distance):
    
            # buffer stop points -> convert their geometries from points into polygon 
            # (we set the radius of polygon = to the radius of sjoin) 
            stops = gpd.GeoDataFrame(stop, geometry=gpd.points_from_xy(stop.lng, stop.lat))
            stops.set_crs('epsg:4326',inplace=True)
            stops.to_crs('epsg:3857',inplace=True)
            stops['geometry_stop'] = stops['geometry']
            stops['geometry'] = stops['geometry_stop'].buffer(max_distance)

            return stops



        def semantic_enrichment(stop, semantic_df, suffix):
        
            #print("DEBUG enrichment occasional stops...")
            
            # duplicate geometry column because we loose it during the sjoin_nearest
            s_df = semantic_df.copy()
            
            # Filter out the POIs without a name!
            s_df = s_df.loc[s_df['name'].notna(), :]
            
            s_df['geometry_'+suffix] = s_df['geometry']
            s_df['element_type'] = s_df['element_type'].astype(str)
            s_df['osmid'] = s_df['osmid'].astype(str)
            
            #print(f"Stampa df stop occasionali: {stop.info()}")
            print(f"Stampa df POIs: {s_df.info()}")
            
            # now we can use sjoin_nearest to obtain the results we want
            mats = stop.sjoin_nearest(s_df, max_distance=0.00001, how='left', rsuffix=suffix)

            # Remove the POIs that have been associated with the same stop multiple times.
            mats.drop_duplicates(subset=['stop_id', 'osmid'], inplace = True)
            
            # compute the distance between the stop point and the POI geometry
            mats['distance'] = mats['geometry_stop'].distance(mats['geometry_'+suffix])
            
            # sort by distance
            mats = mats.sort_values(['tid','stop_id','distance'])
            
            #print(f"Stampa df risultati: {mats.info()}")
            return mats



        def download_poi_osm(list_pois, place, semantic_granularity) :
        
            # Here we download the POIs from OSM if the list of types of POIs is not empty.
            gdf_ = gpd.GeoDataFrame()
            if list_pois != [] :
            
                for key in list_pois:

                    # downloading POI
                    print(f"Downloading {key} POIs from OSM...")
                    poi = ox.geometries_from_place(place, tags={key:True})
                    print(f"Download completed! Dataframe with the downloaded POIs: {poi}")
                    
                    # Immediately return the empty dataframe if it doesn't contain any suitable POI...
                    if poi.empty : 
                        print("No POI found...")
                        new_cols = ['osmid', 'element_type', 'name', 'wikidata', 'geometry', 'category']
                        poi = poi.reindex(poi.columns.union(new_cols), axis=1)
                        return poi
                    
                    
                    # convert list into string in order to save poi into parquet
                    if 'nodes' in poi.columns:
                        poi['nodes'] = poi['nodes'].astype(str)

                    if 'ways' in poi.columns:
                        poi['ways'] = poi['ways'].astype(str)

                    poi.reset_index(inplace=True)
                    poi.rename(columns={key: 'category'}, inplace=True)
                    poi['category'].replace({'yes': key}, inplace=True)

                    if poi.crs is None:
                        poi.set_crs('epsg:4326',inplace=True)
                        poi.to_crs('epsg:3857',inplace=True)
                    else:
                        poi.to_crs('epsg:3857',inplace=True)
                    
                    # Now drop the columns with too many missing values...
                    poi = select_columns(poi, semantic_granularity)
                    
                    # Now write out this subset of POIs to a file. 
                    poi.to_parquet('data/poi/' + key + '.parquet')

                    # And finally, concatenate this subset of POIs to the other POIs
                    # that have been added to the main dataframe so far.
                    gdf_ = pd.concat([gdf_, poi])
            
            
                # print(f"A few info on the POIs downloaded from OSM: {gdf_.info()}")
                gdf_.to_parquet('data/poi/pois.parquet')
                return gdf_



        print("Executing occasional stop augmentation with POIs...")
        print(self.systematic)
        self.occasional = stops[~stops['stop_id'].isin(self.systematic['stop_id'])]
        

        # Get a POI dataset, either from OSM or from a file. 
        gdf_ = None
        if self.path_poi is None :
            print(f"Downloading POIs from OSM for the location of {self.poi_place}. Selected types of POIs: {self.list_pois}")
            gdf_ = download_poi_osm(self.list_pois, self.poi_place, self.semantic_granularity)
        else :
            gdf_ = self.path_poi
            print(f"Using a POI file: {gdf_}!")

            if gdf_.crs is None:
                gdf_.set_crs('epsg:4326', inplace=True)
                gdf_.to_crs('epsg:3857', inplace=True)
            else:
                gdf_.to_crs('epsg:3857', inplace=True)
        print(f"A few info on the POIs that will be used to enrich the occasional stops: {gdf_.info()}")


        # Calling functions internal to this method...
        o_stops = preparing_stops(self.occasional, self.max_distance)
        mat = semantic_enrichment(o_stops, gdf_[['element_type','osmid','name','wikidata','geometry','category']], 'poi')

        ######## PROVA ###########
        # mat.set_index(['stop_id','lat','lng'],inplace=True)
        self.mats = mat.copy()
        self.mats.to_parquet('data/enriched_occasional.parquet')
        
        
        
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

            return(weather_enrichment)
        
            
            
        df_weather_enrichment = None
        print(f"Valore self.weather: {self.weather}")
        if self.weather :
            print("Adding weather info to the trajectories...")

            traj_cleaned = pd.read_parquet('./data/temp_dataset/traj_cleaned.parquet')
            weather = self.upload_weather
            
            df_weather_enrichment = weather_enrichment(traj_cleaned, weather)
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
            traj_cleaned = pd.read_parquet('data/temp_dataset/traj_cleaned.parquet')
            builder.add_trajectories(traj_cleaned)

            # Add to the RDF graph the stops, the moves, and the semantic information 
            # associated with the trajectories (social media posts, weather), the stops (type of stop, POIs),
            # and the moves (transportation mean).
            builder.add_occasional_stops(self.mats)
            builder.add_systematic_stops(self.systematic)
            builder.add_moves(self.moves)
            
            # Add weather information to the trajectories.
            if df_weather_enrichment is not None :
                builder.add_weather(df_weather_enrichment)
                
            # Add weather information to the trajectories.
            if tweets_RDF is not None :
                builder.add_social(tweets_RDF)
            
            # Output the RDF graph to disk in Turtle format.
            print("Saving the KG to disk!")
            builder.serialize_graph('kg.ttl')

        print("Enrichment complete!")
           
    def get_results(self) -> dict:
        return {'moves' : self.moves,
                'occasional' : self.occasional,
                'systematic' : self.systematic,
                'mats' : self.mats,
                'tweets' : self.tweets}

    def get_params_input(self) -> list[str] :
        return ['move_enrichment',
                'place',
                'poi_categories',
                'path_poi',
                'max_dist',
                'social_enrichment',
                'weather_enrichment',
                'create_rdf']

    def get_params_output(self) -> list[str] :
        return ['moves', 'occasional', 'systematic', 'mats', 'tweets']
        
    def reset_state(self) :
        pass