import pandas as pd
import geopandas as gpd
import os

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL

import plotly.express as px
import plotly.graph_objects as go

from core.InteractiveModuleInterface import InteractiveModuleInterface
from core.InteractivePipeline import InteractivePipeline
from .Enrichment import Enrichment


class InteractiveEnrichment(InteractiveModuleInterface):
    '''
    `InteractiveEnrichment` is a class that models the semantic enrichment module. This class allows to:
    1) enrich moves with transportation mean
    2) enrich stops labeling them as occasional and systematic ones
        2.a) occasional stops are enriched with PoIs, weather, etc.
        2.b) systematic stops are enriched as home/work or other
    '''


    ### STATIC FIELDS ###
    
    id_class = 'Enrichment'



    ### CLASS CONSTRUCTOR ###
    
    def __init__(self, app : Dash, pipeline : InteractivePipeline) :

        self.enrich_moves = None
        self.app = app
        self.pipeline = pipeline
        self.prev_module = None

        self.enrichment = Enrichment()
        self.results_enrichment = None
    
    
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
            State(component_id=self.id_class + '-geohash_precision', component_property='value'),
            State(component_id=self.id_class + '-systematic_threshold', component_property='value'),
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
    
    def register_prev_module(self, prev_module : InteractiveModuleInterface) :
        
        print(f"Registering prev module {prev_module} in module {self.id_class}")
        self.prev_module = prev_module
        

    def populate_input_area(self) :
        
        web_components = []
        
        
        if (self.prev_module is None) or (next(iter(self.prev_module.get_results().values())) is None) :
            web_components.append(html.H5(children = f"No stop and moves data available!"))
            web_components.append(html.H5(children = f"Please, execute the segmentation module first!"))
        
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


            web_components.append(html.H5(children="Systematic stop detection"))
            web_components.append(html.Span(children="Insert the geohash precision (this determines the cell size): "))
            web_components.append(dcc.Input(id=self.id_class + '-geohash_precision',
                                            value=7,
                                            type='number',
                                            placeholder='Insert geohash precision...'))
            web_components.append(html.Br())
            web_components.append(html.Span(children="Insert the cell stop-counter threshold (used to find the systematic stops): "))
            web_components.append(dcc.Input(id=self.id_class + '-systematic_threshold',
                                            value=5,
                                            type='number',
                                            placeholder='Insert threshold...'))
            web_components.append(html.Br())
            web_components.append(html.Br())
            
            # Input social media posts enrichment
            web_components.append(html.H5(children = "Enrich trajectory users with social media posts: "))
            web_components.append(html.Span(children = "Path to file containing the posts (write 'no' if no enrichment should be done): "))
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
                              poi_place,
                              poi_categories,
                              path_poi,
                              max_dist,
                              geohash_precision,
                              systematic_threshold,
                              social_enrichment,
                              weather_enrichment,
                              create_rdf,
                              button_state):
        
        outputs = []
        if button_state is not None :
        
            print(f"Executing function get_input_and_execute of module {self.id_class}! {button_state}")


            # Check input correctness.
            if [x for x in (move_enrichment, poi_place, poi_categories, path_poi, max_dist, geohash_precision,
                            systematic_threshold, social_enrichment, weather_enrichment, create_rdf) if x is None]:
                outputs.append(html.H6(children='Error: some input values were not provided!'))
                return None, outputs

            poi_df = None
            if (path_poi != 'no') and (os.path.isfile(path_poi) is False) :
                outputs.append(html.H6(children='Error: invalid path to the poi file!'))
                return None, outputs
            else :
                poi_df = gpd.read_parquet(path_poi)

            social_df = None
            if (social_enrichment != 'no') and (os.path.isfile(social_enrichment) is False) :
                outputs.append(html.H6(children='Error: invalid path to the social media file!'))
                return None, outputs
            else :
                social_df = pd.read_parquet(social_enrichment) if social_enrichment != 'no' else None

            weather_df = None
            if (weather_enrichment != 'no') and (os.path.isfile(weather_enrichment) is False) :
                outputs.append(html.H6(children='Error: invalid path to the weather file!'))
                return None, outputs
            else :
                weather_df = pd.read_parquet(weather_enrichment) if weather_enrichment != 'no' else None


            # Esegui il core dell'istanza.
            prev_results = self.prev_module.get_results()
            self.enrich_moves = True if move_enrichment == 'yes' else False
            dic_params = {'moves' : prev_results['moves'],
                          'move_enrichment' : self.enrich_moves,
                          'stops' : prev_results['stops'],
                          'poi_place' : poi_place,
                          'poi_categories' : None if poi_categories == ['no'] else poi_categories,
                          'path_poi' : poi_df,
                          'max_dist' : max_dist,
                          'geohash_precision' : geohash_precision,
                          'systematic_threshold' : systematic_threshold,
                          'social_enrichment' : social_df,
                          "weather_enrichment" : weather_df,
                          'create_rdf' : True if create_rdf == 'yes' else False}
            self.enrichment.execute(dic_params)
            self.results_enrichment = self.enrichment.get_results()
            
            
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


    def show_trajectories(self, user) :

        options = []
        if user is not None:
            trajectories = self.get_trajectories(user)
            systematic = self.results_enrichment['systematic']
            occasional = self.results_enrichment['occasional']

            list_traj = []
            for i in trajectories :
                num_sys = str(len(systematic[systematic['tid']==i]))
                num_occ = str(len(occasional[occasional['tid']==i]))
                text = i + ' (systematic: ' + num_sys + ', occasional: ' + num_occ + ')'
                list_traj.append({'label': text, 'value': i})

            options.extend([html.H6(children='Trajectory plotter (choose one from the dropdown menu):',
                                    style={'font-weight': 'bold'}),
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
        
        
        # Display transportation means information, if the moves have been enriched.
        if self.enrich_moves :
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


        ### Plot the systematic and occasional stops. ###
        occasional = self.results_enrichment['enriched_occasional']
        systematic = self.results_enrichment['systematic']

        ### Plot the systematic stops. ###
        mats_systematic = systematic[systematic['uid'] == user].copy()
        if len(mats_systematic) :
            outputs.append(html.H6(children='Overall distribution of the systematic stops:',
                                   style={'font-weight': 'bold'}))

            ### Preparing the information concerning the systematic stops ###
            mats_systematic['systematic_id'] = mats_systematic['systematic_id'].astype(str)
            mats_systematic['home'] = round((mats_systematic['home'] * 100), 2).astype(str)
            mats_systematic['work'] = round((mats_systematic['work'] * 100), 2).astype(str)
            mats_systematic['other'] = round((mats_systematic['other'] * 100), 2).astype(str)
            mats_systematic['importance'] = round((mats_systematic['importance'] * 100), 2).astype(str)
            mats_systematic['frequency'] = mats_systematic['frequency'].astype(str)
            mats_systematic['start'] = mats_systematic['datetime'].astype(str)
            mats_systematic['duration'] = (mats_systematic['leaving_datetime'] - mats_systematic['datetime']).astype(str)
            mats_systematic['weekday'] = mats_systematic['datetime'].dt.weekday.astype(str)

            mats_systematic['description'] = '</br><b>Systematic ID</b>: ' + mats_systematic['systematic_id'] \
                                             + '</br><b>Start time</b>: ' + mats_systematic['start'] \
                                             + '</br><b>Day of the week</b>: ' + mats_systematic['weekday'] \
                                             + '</br><b>Duration</b>: ' + mats_systematic['duration'] \
                                             + '</br><b>Home</b>: ' + mats_systematic['home'] \
                                             + '%</br><b>Work</b>: ' + mats_systematic['work'] \
                                             + '%</br><b>Other</b>: ' + mats_systematic['other'] \
                                             + '%</br><b>Importance</b>: ' + mats_systematic['importance'] \
                                             + '%</br><b>Frequency </b>: ' + mats_systematic['frequency']
            systematic_desc = list(mats_systematic['description'])

            fig = go.Figure(go.Scattermapbox(mode="markers", name='systematic stops',
                                             lon=mats_systematic.lng,
                                             lat=mats_systematic.lat,
                                             text=systematic_desc,
                                             hoverinfo='text',
                                             marker={'size': 10, 'color': '#2BD98C'}))
            fig.update_layout(mapbox_style = "open-street-map",
                              margin = {"r" : 0, "t" : 0 ,"l" : 0, "b" : 0},
                              mapbox = dict(center = dict(lat=mats_systematic.iloc[0].lat,lon=mats_systematic.iloc[0].lng),
                                            zoom=10))
            outputs.append(dcc.Graph(figure = fig))


        ### Plot the occasional stops. ###
        mats_stops = occasional[occasional['uid'] == user].copy()
        if len(mats_stops) :
            outputs.append(html.H6(children='Overall distribution of the occasional stops:',
                                   style={'font-weight': 'bold'}))

            mats_stops['distance'] = round(mats_stops['distance'], 2)
            mats_stops['description'] = '</br><b>PoI category</b>: ' + \
                                        mats_stops['category'] + \
                                        ' <b>Distance</b>: ' + \
                                        mats_stops['distance'].astype(str)

            limit_pois = 8
            gb_occ_stops = mats_stops.groupby('stop_id')
            matched_lat = gb_occ_stops['lat'].first().tolist()
            matched_lng = gb_occ_stops['lng'].first().tolist()
            matched_pois = []
            for key, item in gb_occ_stops:
                tmp = item['description']
                size = tmp.shape[0]
                limit = min(size, limit_pois)

                stringa = ''
                if ~item['distance'].isna().all():
                    tmp = tmp.head(limit)
                    stringa = tmp.str.cat(sep="")
                    if size > limit_pois:
                        stringa = stringa + f"</br>(...and other {size - limit_pois} POIs)"
                else:
                    stringa = 'No POI could be associated with this occasional stop!'

                matched_pois.append(stringa)

            fig2 = go.Figure(go.Scattermapbox(mode="markers", name='occasional stops',
                                              lon = matched_lng,
                                              lat = matched_lat,
                                              text = matched_pois,
                                              hoverinfo='text',
                                              marker={'size': 10, 'color': '#F14C2B'}))
            fig2.update_layout(mapbox_style="open-street-map",
                              margin={"r": 0, "t": 0, "l": 0, "b": 0},
                              mapbox=dict(center=dict(lat = matched_lat[0],
                                                      lon = matched_lng[0]),
                                          zoom=10))
            outputs.append(dcc.Graph(figure=fig2))


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
        mats_moves, mats_stops, mats_systematic = self.get_enriched_stop_move(user, traj)


        ### Preparing the information concerning the moves ###

        # print(f"DEBUG PLOT MOVES: {mats_moves}")
        mats_moves['label'] = mats_moves['label'].map(transport) if self.enrich_moves else 'NA'
        fig = px.line_mapbox(mats_moves,
                             lat="lat",
                             lon="lng",
                             color="tid",
                             hover_data=["label","temperature","w_conditions"],
                             labels={"label":"transportation mean", "w_conditions":"weather condition"})
        
        
        
        ### Prepare the information concerning the occasional stops... ###

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

        mats_systematic['systematic_id'] = mats_systematic['systematic_id'].astype(str)
        mats_systematic['home'] = round((mats_systematic['home']*100),2).astype(str)
        mats_systematic['work'] = round((mats_systematic['work']*100),2).astype(str)
        mats_systematic['other'] = round((mats_systematic['other']*100),2).astype(str)
        mats_systematic['importance'] = round((mats_systematic['importance'] * 100), 2).astype(str)
        mats_systematic['frequency'] = mats_systematic['frequency'].astype(str)
        mats_systematic['start'] = mats_systematic['datetime'].astype(str)
        mats_systematic['duration'] = (mats_systematic['leaving_datetime'] - mats_systematic['datetime']).astype(str)
        mats_systematic['weekday'] = mats_systematic['datetime'].dt.weekday.astype(str)

        mats_systematic['description'] = '</br><b>Systematic ID</b>: ' + mats_systematic['systematic_id'] \
                                         + '</br><b>Start time</b>: ' + mats_systematic['start'] \
                                         + '</br><b>Day of the week</b>: ' + mats_systematic['weekday'] \
                                         + '</br><b>Duration</b>: ' + mats_systematic['duration']\
                                         + '</br><b>Home</b>: ' + mats_systematic['home']\
                                         + '%</br><b>Work</b>: ' + mats_systematic['work']\
                                         + '%</br><b>Other</b>: ' + mats_systematic['other'] \
                                         + '%</br><b>Importance</b>: ' + mats_systematic['importance'] \
                                         + '%</br><b>Frequency </b>: ' + mats_systematic['frequency']
        systematic_desc = list(mats_systematic['description'])

        fig.add_trace(go.Scattermapbox(mode = "markers", name = 'systematic stops',
                                       lon = mats_systematic.lng,
                                       lat = mats_systematic.lat,
                                       text = systematic_desc,
                                       hoverinfo = 'text',
                                       marker = {'size': 10,'color': '#2BD98C'}))
        
        
        
        ### Setting the figure's last parameters... ###
        
        fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=12,
                          margin={"r":0,"t":0,"l":0,"b":0})
        fig.update_traces(line=dict(color='#2B37F1',width=2))

        return dcc.Graph(figure=fig)        
           
           
    def get_results(self) :
        return self.enrichment.get_results()
        
        
    def reset_state(self) :
        print(f"Resetting state of the module {self.id_class}")
        self.enrichment.reset_state()
                     
        
    def get_users(self):
        moves = self.results_enrichment['moves'].copy()
        moves.reset_index(inplace=True)
        return moves['uid'].unique()


    def get_trajectories(self, uid):
        moves = self.results_enrichment['moves'].copy()
        return moves[moves['uid']==uid]['tid'].unique()


    def get_systematic(self, uid):
        systematic = self.results_enrichment['systematic'].copy()
        return len(systematic[systematic['uid']==uid])


    def get_occasional(self, uid):
        occasional = self.results_enrichment['occasional'].copy()
        return len(occasional[occasional['uid']==uid])


    def get_transport_duration(self,uid):

        moves = self.results_enrichment['moves'].copy()

        first_transport = moves[moves['uid']==uid].groupby(['label','tid']).first()['datetime']
        last_transport = moves[moves['uid']==uid].groupby(['label','tid']).last()['datetime']

        duration_tid = last_transport - first_transport
        
        duration = pd.DataFrame(duration_tid.groupby('label').sum())
        duration.reset_index(inplace=True)

        return duration


    def get_tweets(self,uid):

        if self.results_enrichment['tweets'] is not None :
            tweets = self.results_enrichment['tweets']
            return tweets[tweets['uid']==uid]['text'].unique()
        else :
            return []


    def get_enriched_stop_move(self, uid, traj_id):

        moves = self.results_enrichment['moves'].copy()
        enriched_occasional = self.results_enrichment['enriched_occasional'].copy()
        systematic = self.results_enrichment['systematic'].copy()

        return moves[(moves['uid']==uid) & (moves['tid'] == traj_id)],\
               enriched_occasional[(enriched_occasional['uid']==uid) & (enriched_occasional['tid']==traj_id)],\
               systematic[(systematic['uid']==uid) & (systematic['tid']==traj_id)]