import geopandas as gpd
import pandas as pd
import numpy as np
import math

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.kinematic_features import KinematicFeatures as spatial

import pickle
from sklearn.preprocessing import StandardScaler
import geohash2
import osmnx as ox

from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL

import plotly.express as px
import plotly.graph_objects as go

from core.ModuleInterface import ModuleInterface
from core.RDF_builder import RDFBuilder



class Enrichment(ModuleInterface):
    '''
    `stop_move_enrichment` is a class that models the semantic enrichment module. This class allows to:
    1) enrich moves with transportation mean
    2) enrich stops labeling them as occasional and systematic ones
        2.a) occasional stops are enriched with PoIs, weather, etc.
        2.b) systematic stops are enriched as home/work or other
    '''


    ### STATIC FIELDS ###
    
    id_class = 'Enrichment'



    ### CLASS CONSTRUCTOR ###
    
    def __init__(self, app, pipeline) :
    
        self.app = app
        self.pipeline = pipeline
        self.prev_module = None
    
    
        ### Here we define and register all the callbacks that must be managed by the instance of this class ###
        
        # This is the callback associated with the input area.
        self.app.callback \
        (
            Output(component_id = 'loading-' + self.id_class + '-c', component_property='children'),
            Output(component_id = 'output-' + self.id_class, component_property='children'),
            State(component_id = self.id_class + '-move_en', component_property='value'),
            State(component_id = self.id_class + '-place', component_property='value'),
            State(component_id = self.id_class + '-poi_cat', component_property='value'),
            State(component_id = self.id_class + '-poi_file', component_property='value'),
            State(component_id = self.id_class + '-max_dist', component_property='value'),
            State(component_id = self.id_class + '-social_en', component_property='value'),
            State(component_id = self.id_class + '-weather_en', component_property='value'),
            State(component_id = self.id_class + '-write_rdf', component_property='value'),
            Input(component_id = self.id_class + '-run', component_property='n_clicks')
        )(self.get_input_and_execute)
        
        
        # This is the callback in charge of creating a dropdown menu containing the trajectories
        # associated with a user.
        self.app.callback\
        ( 
            Output(component_id = 'traj-' + self.id_class, component_property = 'children'),
            Input(component_id = 'user_sel-' + self.id_class, component_property = 'value'),
        )(self.show_trajectories)
        
        
        # This is the callback in charge of showing summary information concerning a trajectory.
        self.app.callback\
        (
            Output(component_id = 'user_info-' + self.id_class, component_property = 'children'),
            Input(component_id = 'user_sel-' + self.id_class, component_property = 'value')
        )(self.info_user)
        
        
        # This is the callback in charge of plotting a trajectory.
        self.app.callback\
        (
            Output(component_id = 'traj_display-' + self.id_class, component_property = 'children'),
            State(component_id = 'user_sel-' + self.id_class, component_property='value'),
            Input(component_id = 'traj_sel-' + self.id_class, component_property='value')
        )(self.display_user_trajectory)
        
    
    
    ### CLASS PUBLIC METHODS ###
    
    def register_prev_module(self, prev_module) :
        
        print(f"Registering prev module {prev_module} in module {self.id_class}")
        self.prev_module = prev_module
        

    def populate_input_area(self) :
        
        web_components = []
        
        
        if self.prev_module.get_results() is None :
            web_components.append(html.H5(children = f"No data available from the {self.prev_module.id_class} module!"))
            web_components.append(html.H5(children = f"Please, execute it first!"))
        
        else :
            # Input move enrichment with transportation means 
            web_components.append(html.H5(children = "Move enrichment"))
            web_components.append(html.Span(children = "Add transportation means to moves?"))
            web_components.append(dcc.Dropdown(id = self.id_class + '-move_en',
                                               options = [{"label": "yes", "value": "yes"},
                                                          {"label":"no","value":"no"}],
                                               value = "yes",
                                               style={'color':'#333'}))
            web_components.append(html.Br())
            
            
            # Input stop enrichment with POIs 
            web_components.append(html.H5(children = "Add POIs to occasional stops"))
            web_components.append(html.Span(children = "Insert the name of the city (to download PoIs from OpenStreetMap): "))
            web_components.append(dcc.Input(id = self.id_class + '-place',
                                            value = "Rome, Italy",
                                            type = 'text',
                                            placeholder = 'Insert city...'))
            web_components.append(html.Br())                                
            web_components.append(html.Span(children = "PoI categories (considered only when downloading from OpenStreetMap)"))
            web_components.append(dcc.Dropdown(id = self.id_class + '-poi_cat',
                                               options = [{"label": "amenity", "value": "amenity"},
                                                          {"label": "aeroway", "value": "aeroway"},
                                                          {"label": "building", "value": "building"},
                                                          {"label": "historic", "value": "historic"},
                                                          {"label": "healthcare", "value": "healthcare"},
                                                          {"label": "landuse", "value": "landuse"},
                                                          {"label": "office", "value": "office"},
                                                          {"label": "public_transport", "value": "public_transport"},
                                                          {"label": "shop", "value": "shop"},
                                                          {"label": "tourism", "value": "tourism"},
                                                          {"label":"no enrichment","value":"no"}],
                                               value = ["amenity"],
                                               multi = True,
                                               style = {'color':'#333'}))
                                               
            web_components.append(html.Span(children = "... or upload your file containing a POI dataset (leave 'no' if no file is uploaded) "))
            web_components.append(dcc.Input(id = self.id_class + '-poi_file',
                                            value = "./data/Rome/poi/pois.parquet",
                                            type = 'text',
                                            placeholder = 'Path to the POI dataset...'))   
            web_components.append(html.Br())
            
            
            web_components.append(html.Span(children = "Maximum distance from the centroid of the stops (in meters): "))
            web_components.append(dcc.Input(id = self.id_class + '-max_dist',
                                            value = 100,
                                            type = 'number',
                                            placeholder = 'Max distance from PoIs (in meters)...'))
            web_components.append(html.Br())
            web_components.append(html.Br())
            
            
            # Input social media posts enrichment
            web_components.append(html.H5(children = "Enrich trajectory users with social media posts: "))
            web_components.append(html.Span(children = "Path to file containing the posts (write 'no' if no enrichment should be done: "))
            web_components.append(dcc.Input(id = self.id_class + '-social_en',
                                            value = './data/tweets/tweets_rome.parquet',
                                            type = 'text',
                                            placeholder = 'Path to file containing the posts...'))
            web_components.append(html.Br())
            web_components.append(html.Br())
            
            
            # Input weather information enrichment
            web_components.append(html.H5(children = "Enrich trajectories with weather information: "))
            web_components.append(html.Span(children = "Path to file containing the posts (write 'no' if no enrichment should be done):"))
            web_components.append(dcc.Input(id = self.id_class + '-weather_en',
                                            value = './data/weather/weather_conditions.parquet',
                                            type = 'text',
                                            placeholder = 'Path to file containing weather information...'))
            web_components.append(html.Br())
            web_components.append(html.Br())
            
            
            # Input RDF graph
            web_components.append(html.H5(children = "Save the enriched trajectories into an RDF graph: "))
            web_components.append(dcc.Dropdown(id = self.id_class + '-write_rdf',
                                               options = [{"label": "yes", "value": "yes"},
                                                          {"label":"no","value":"no"}],
                                               value = "yes",
                                               style={'color':'#333'}))
            web_components.append(html.Br())
            web_components.append(html.Br())        
                                       
            web_components.append(html.Button(id = self.id_class + '-run', children='RUN'))           
        
        return web_components
        
        
    def get_input_and_execute(self,
                              move_enrichment,
                              place,
                              poi_categories,
                              path_poi,
                              max_dist,
                              social_enrichment,
                              weather_enrichment,
                              create_rdf,
                              button_state):
        
        outputs = []        
        if button_state is not None :
        
            print(f"Esecuzione get_input_and_execute del modulo {self.id_class}! {button_state}")
        
            ### Reset the state of the static variables...
            self.stops = pd.DataFrame()
            self.moves = pd.DataFrame()
            self.mats = gpd.GeoDataFrame()


            # Input parsing...
            if move_enrichment == 'yes':
                self.enrich_moves = True
                self.moves = pd.read_parquet('data/temp_dataset/moves.parquet')
            else:
                self.enrich_moves = False

            self.place = place

            if poi_categories == ['no']:
                self.list_pois = []
            else:
                self.list_pois = poi_categories

            if path_poi == 'no':
                self.upload_stops = 'no'
            else:
                self.upload_stops = path_poi

            self.semantic_granularity = 80
            self.max_distance = max_dist

            # Variabili gestione arricchimento social media post.
            if social_enrichment == 'no':
                self.tweet_user = False
            else:
                self.tweet_user = True
                self.upload_users = social_enrichment

            # Variabili gestione arricchimento weather information.
            if weather_enrichment == 'no':
                self.weather = False
            else:
                self.weather = True
                self.upload_trajs = weather_enrichment
            
            # Variabili gestione scrittura grafo RDF.
            self.rdf = 'yes' if create_rdf == 'yes' else 'no'
            
            
            # Esegui il core dell'istanza.
            self.core()
            
            
            # Inizializza il dropdown con la lista di utenti da mostrare nell'area di output dell'interfaccia web.
            list_users = [{'label': u, 'value': u} for u in self.get_users()]
            outputs.append(html.Div(id='users-' + self.id_class,
                                    children = [html.P(children = 'User:'),
                                                dcc.Dropdown(id = 'user_sel-' + self.id_class,
                                                             options = list_users,
                                                             style={'color':'#333'}),
                                                html.Br(),
                                                html.Div(id = 'user_info-' + self.id_class),
                                                html.Br(),
                                                html.Div(id = 'traj-' + self.id_class)]))
            
            
        # Ritorna gli output finali per l'interfaccia web.
        return None, outputs


    def core(self):
    
        print("Executing the core of the semantic enrichment module...")


        #################################
        ### ---- MOVE ENRICHMENT ---- ###
        #################################

        moves = self.moves
        if self.enrich_moves == True:
            print("Executing move enrichment...")
            
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
            
            self.moves = moves
            moves.to_parquet('data/enriched_moves.parquet')



        ############################################
        ### ---- SYSTEMATIC STOP ENRICHMENT ---- ###
        ############################################
        
        print("Executing systematic stop enrichment...")

        self.stops = pd.read_parquet('data/temp_dataset/stops.parquet')

        stops = self.stops.copy()
        stops.reset_index(inplace=True)
        stops.rename(columns={'index':'stop_id'},inplace=True)
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
        systematic_stops = sp[sp['frequency'] > 2]
        
        # 1 - case where systematic stops have been found...
        if(systematic_stops.shape[0] != 0) :
            systematic_stops['start_time'] = systematic_stops['datetime'].dt.hour
            systematic_stops['end_time'] = systematic_stops['leaving_datetime'].dt.hour
            systematic_stops.reset_index(inplace=True,drop=True)

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
            largest_index = largest.index.droplevel(0)
            freq['home'] = 0
            freq['work'] = 0
            freq['other'] = 0

            w_home = [0.6,0.1,0.1,0.4]
            w_work = [0.1,0.6,0.4,0.1]
        
            freq['p_home'] = (freq[['night','morning','afternoon','evening']].loc[largest_index] * w_home).sum(axis=1)
            freq['p_work'] = (freq[['night','morning','afternoon','evening']].loc[largest_index] * w_work).sum(axis=1)
            freq['home'] = freq['p_home'] / freq[['p_home','p_work']].sum(axis=1)
            freq['work'] = freq['p_work'] / freq[['p_home','p_work']].sum(axis=1)
            freq['other'].loc[~freq.index.isin(largest_index)] = 1
            freq['home'].fillna(0,inplace=True)
            freq['work'].fillna(0,inplace=True)

            systematic_stops.set_index(['uid','pos_hashed'],inplace=True)
            systematic_stops['home'] = 0
            systematic_stops['work'] = 0
            systematic_stops['other'] = 0
            systematic_stops['home'].loc[systematic_stops.index.isin(freq.index)] = freq['home'].loc[freq.index.isin(systematic_stops.index)]
            systematic_stops['work'].loc[systematic_stops.index.isin(freq.index)] = freq['work'].loc[freq.index.isin(systematic_stops.index)]
            systematic_stops['other'].loc[systematic_stops.index.isin(freq.index)] = freq['other'].loc[freq.index.isin(systematic_stops.index)]
            systematic_stops.reset_index(inplace=True)
            
        # 2 - case where no systematic stops have been found...adapt the empty dataframe
        #     for the code that will use it.
        else :
            new_cols = ['home', 'work', 'other', 'start_time', 'end_time']
            systematic_stops = systematic_stops.reindex(systematic_stops.columns.union(new_cols), axis=1)
            
        # Write out the dataframe...
        self.systematic = systematic_stops
        self.systematic.to_parquet('data/systematic_stops.parquet')
        
        
        
        ############################################
        ### ---- OCCASIONAL STOP ENRICHMENT ---- ###
        ############################################

        def select_columns(gdf, threshold=80.0):
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
            s_df['geometry_'+suffix] = s_df['geometry']
            s_df['osmid'] = s_df['osmid'].astype(str)
            
            #print(f"Stampa df stop occasionali: {stop.info()}")
            #print(f"Stampa df POIs: {s_df.info()}")
            
            # now we can use sjoin_nearest to obtain the results we want
            mats = stop.sjoin_nearest(s_df, max_distance=0.00001, how='left', rsuffix=suffix)
            
            # compute the distance between the stop point and the POI geometry
            #mats['distance_'+suffix] = mats['geometry_stop'].distance(mats['geometry_'+suffix])
            mats['distance'] = mats['geometry_stop'].distance(mats['geometry_'+suffix])
            
            # sort by distance
            #mats = mats.sort_values(['stop_id','distance_'+suffix])
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
                        new_cols = ['osmid', 'wikidata', 'geometry', 'category']
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



        print("Executing occasional stop enrichment...")

        occasional_stops = stops[~stops['stop_id'].isin(systematic_stops['stop_id'])]
        self.occasional = occasional_stops
        

        # Get a POI dataset, either from OSM or from a file. 
        gdf_ = None
        if self.upload_stops == 'no' :
            print(f"Downloading POIs from OSM for the location of {self.place}. Selected types of POIs: {self.list_pois}")
            gdf_ = download_poi_osm(self.list_pois, self.place, self.semantic_granularity)
        else :    
            print(f"Using a POI file: {self.upload_stops}!")
            gdf_ = gpd.read_parquet(self.upload_stops)

            if gdf_.crs is None:
                gdf_.set_crs('epsg:4326', inplace=True)
                gdf_.to_crs('epsg:3857', inplace=True)
            else:
                gdf_.to_crs('epsg:3857', inplace=True)
        print(f"A few info on the POIs that will be used to enrich the occasional stops: {gdf_.info()}")


        # Calling functions internal to this method...
        o_stops = preparing_stops(occasional_stops, self.max_distance)    
        mat = semantic_enrichment(o_stops, gdf_[['osmid','wikidata','geometry','category']], 'poi')

        ######## PROVA ###########
        #mat.set_index(['stop_id','lat','lng'],inplace=True)
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
            weather = pd.read_parquet(self.upload_trajs)
            
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
            
            tweets = pd.read_parquet(self.upload_users)
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
            
            

        ##############################################
        ###            RDF GRAPH SAVE              ###
        ##############################################
        
        if self.rdf == 'yes':
            
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
            builder.serialize_graph('kg.ttl')
    

    def show_trajectories(self, user) :

        options = []
        if user is not None:

            list_traj = [{'label': i, 'value': i} for i in self.get_trajectories(user)]
            options.extend([html.P(children = 'Trajectories:'),
                           dcc.Dropdown(id = 'traj_sel-' + self.id_class,
                                        options = list_traj,
                                        style={'color':'#333'}),
                           html.Br(),
                           html.Div(id = 'traj_display-' + self.id_class)])

        return options


    def info_user(self, user):

        outputs = []


        if user is None:
            return None


        # Display stops information...
        num_systematic = self.get_systematic(user)
        num_occasional = self.get_occasional(user)

        outputs.append(html.H6(children='Aspects concerning the stops:',
                               style={'font-weight':'bold'}))
        outputs.append(html.Span(children='Number of systematic stops: ',
                                 style={'font-weight':'bold'}))
        outputs.append(html.Span(children=str(num_systematic)+' \t'))
        outputs.append(html.Br())
        outputs.append(html.Span(children='Number of occasional stops: ',
                                 style={'font-weight':'bold'}))
        outputs.append(html.Span(children=str(num_occasional)))
        outputs.append(html.Br())
        outputs.append(html.Br())
        
        
        # Display transportation means information...
        duration_transport = self.get_transport_duration(user)
        duration_walk = duration_transport[duration_transport['label']==0]['datetime'].astype(str).values
        duration_bike = duration_transport[duration_transport['label']==1]['datetime'].astype(str).values
        duration_bus = duration_transport[duration_transport['label']==2]['datetime'].astype(str).values
        duration_car = duration_transport[duration_transport['label']==3]['datetime'].astype(str).values
        duration_subway = duration_transport[duration_transport['label']==4]['datetime'].astype(str).values
        duration_train = duration_transport[duration_transport['label']==5]['datetime'].astype(str).values
        duration_taxi = duration_transport[duration_transport['label']==6]['datetime'].astype(str).values

        if len(duration_walk) == 0:
            duration_walk = 0
        else:
            duration_walk = duration_walk[0]

        if len(duration_bike) == 0:
            duration_bike = 0
        else:
            duration_bike = duration_bike[0]

        if len(duration_bus) == 0:
            duration_bus = 0
        else:
            duration_bus = duration_bus[0]

        if len(duration_car) == 0:
            duration_car = 0
        else:
            duration_car = duration_car[0]
        
        if len(duration_subway) == 0:
            duration_subway = 0
        else:
            duration_subway = duration_subway[0]
        
        if len(duration_train) == 0:
            duration_train = 0
        else:
            duration_train = duration_train[0]

        if len(duration_taxi) == 0:
            duration_taxi = 0
        else:
            duration_taxi = duration_taxi[0]
            
        outputs.append(html.H6(children='Aspects concerning the moves (transportation means and duration):',
                               style={'font-weight':'bold'}))
        outputs.append(html.Span(children='Walk: ',style={'font-weight':'bold'}))    
        outputs.append(html.Span(children=str(duration_walk)+' \t'))
        outputs.append(html.Br())
        outputs.append(html.Span(children='Bike: ',style={'font-weight':'bold'}))    
        outputs.append(html.Span(children=str(duration_bike)+' \t'))
        outputs.append(html.Br())
        outputs.append(html.Span(children='Bus: ',style={'font-weight':'bold'}))    
        outputs.append(html.Span(children=str(duration_bus)+' \t'))
        outputs.append(html.Br())
        outputs.append(html.Span(children='Car: ',style={'font-weight':'bold'}))    
        outputs.append(html.Span(children=str(duration_car)+' \t'))
        outputs.append(html.Br())
        outputs.append(html.Span(children='Train: ',style={'font-weight':'bold'}))    
        outputs.append(html.Span(children=str(duration_train)+' \t'))
        outputs.append(html.Br())
        outputs.append(html.Span(children='Subway: ',style={'font-weight':'bold'}))    
        outputs.append(html.Span(children=str(duration_subway)+' \t'))
        outputs.append(html.Br())
        outputs.append(html.Span(children='Taxi: ',style={'font-weight':'bold'}))    
        outputs.append(html.Span(children=str(duration_taxi)+' \t'))
        outputs.append(html.Br())
        outputs.append(html.Br())


        # Display social media information...
        tweets = self.get_tweets(user)
        if len(tweets) != 0:
        
            outputs.append(html.H6(children='Aspects concerning social media:',style={'font-weight':'bold'}))
            children_list = []
            for t in tweets: children_list.append(html.Li(children='Tweet text: ' + str(t)))
            outputs.append(html.Ul(children = children_list))
            outputs.append(html.Br())


        return outputs      


    def display_user_trajectory(self, user, traj):

        # Dictionary holding the transportation modes mapping. 
        transport = { 
            0: 'walk',
            1: 'bike',
            2: 'bus',
            3: 'car',
            4: 'subway',
            5: 'train',
            6: 'taxi'}

        if user is None or traj is None:
            return None

        # Get the dataframes of interest from the enrichment class.
        mats_moves, mats_stops, mats_systematic = self.get_mats(user, traj)


        ### Preparing the information concerning the moves ###

        #print(mats_moves['label'].unique())
        mats_moves['label'] = mats_moves['label'].map(transport)
        fig = px.line_mapbox(mats_moves,
                             lat="lat",
                             lon="lng",
                             color="tid",
                             hover_data=["label","temperature","w_conditions"],
                             labels={"label":"transportation mean", "w_conditions":"weather condition"})
        
        
        
        ### Prepare the information concerning the occasional stops... ###
        mats_stops.drop_duplicates(subset=['category','distance'], inplace = True)
        
        mats_stops['distance'] = round(mats_stops['distance'], 2)
        mats_stops['description'] = '</br><b>PoI category</b>: ' +\
                                    mats_stops['category'] +\
                                    ' <b>Distance</b>: ' +\
                                    mats_stops['distance'].astype(str)
        
        limit_pois = 10
        gb_occ_stops = mats_stops.groupby('stop_id')
        matched_pois = []
        for key, item in gb_occ_stops:
            
            tmp = item['description']
            size = tmp.shape[0]
            limit = min(size, limit_pois)
            
            stringa = ''
            if ~item['distance'].isna().all() :
                tmp = tmp.head(limit)
                stringa = tmp.str.cat(sep = "")
                if size > limit_pois : 
                    stringa = stringa + f"</br>(...and other {size - limit_pois} POIs)"
            else :          
                stringa = 'No POI could be associated with this occasional stop!'
                
            matched_pois.append(stringa)


        fig.add_trace(go.Scattermapbox(mode = "markers", name = 'occasional stops',
                                       lon = mats_stops.lng.unique(),
                                       lat = mats_stops.lat.unique(),
                                       text = matched_pois,
                                       hoverinfo = 'text',
                                       marker = {'size': 10, 'color': '#F14C2B'}))



        ### Preparing the information concerning the systematic stops ###

        mats_systematic['home'] = round((mats_systematic['home']*100),2).astype(str)
        mats_systematic['work'] = round((mats_systematic['work']*100),2).astype(str)
        mats_systematic['other'] = round((mats_systematic['other']*100),2).astype(str)
        mats_systematic['frequency'] = mats_systematic['frequency'].astype(str)

        mats_systematic['description'] = '<b> Home </b>: ' + mats_systematic['home'] + '% </br></br> <b> Work </b>: ' + mats_systematic['work'] + '% </br> <b> Other </b>: ' + mats_systematic['other'] + '% </br> <b> Frequency </b>: ' + mats_systematic['frequency']
        systematic_desc = list(mats_systematic['description'])

        fig.add_trace(go.Scattermapbox(mode = "markers", name = 'systematic stops',
                                       lon = mats_systematic.lng,
                                       lat = mats_systematic.lat,
                                       text = systematic_desc,
                                       hoverinfo = 'text',
                                       marker = {'size': 10,'color': '#2BD98C'}))
        
        
        
        ### Setting the last parameters... ###
        
        fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=12,
                          margin={"r":0,"t":0,"l":0,"b":0})
        fig.update_traces(line=dict(color='#2B37F1',width=2))

        return dcc.Graph(figure=fig)        
           
           
    def get_results(self) :
        return None
        
        
    def reset_state(self) :
        print(f"Resetting state of the module {self.id_class}")
                     
        
    def get_users(self):
        self.moves.reset_index(inplace=True)
        return self.moves['uid'].unique()


    def get_trajectories(self,uid):
        return self.moves[self.moves['uid']==uid]['tid'].unique()


    def get_systematic(self,uid):
        return len(self.systematic[self.systematic['uid']==uid])


    def get_occasional(self,uid):
        return len(self.occasional[self.occasional['uid']==uid])


    def get_transport_duration(self,uid):

        first_transport = self.moves[self.moves['uid']==uid].groupby(['label','tid']).first()['datetime']
        last_transport = self.moves[self.moves['uid']==uid].groupby(['label','tid']).last()['datetime']

        duration_tid = last_transport - first_transport
        
        duration = pd.DataFrame(duration_tid.groupby('label').sum())
        duration.reset_index(inplace=True)

        return duration


    def get_tweets(self,uid):
   
        if self.tweets is not None :
            return self.tweets[self.tweets['uid']==uid]['text'].unique()
        else : return []


    def get_mats(self,uid,traj_id):
        #print(self.mats[self.mats['tid']==traj_id])

        return self.moves[(self.moves['uid']==uid)&(self.moves['tid']==traj_id)], self.mats[(self.mats['uid']==uid)&(self.mats['tid']==traj_id)], self.systematic[(self.systematic['uid']==uid)&(self.systematic['tid']==traj_id)]