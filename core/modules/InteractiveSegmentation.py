import geopandas as gpd
import pandas as pd
import numpy as np

import skmob
from skmob.preprocessing import detection

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL

from core.InteractiveModuleInterface import InteractiveModuleInterface


class InteractiveSegmentation(InteractiveModuleInterface):
    '''
    `stops_and_moves` models a class which instances segment trajectories according to the stop and move paradigm.
    '''
    
    
    ### STATIC FIELDS ###
    
    id_class = 'Segmentation'
    


    ### CLASS CONSTRUCTOR ###
    
    def __init__(self, app : Dash, pipeline : Pipeline) :
        
        self.app = app
        self.pipeline = pipeline
        self.prev_module = None
        
        
        self.stops = None
        self.moves = None
        self.path_pre_traj = './data/temp_dataset/traj_cleaned.parquet'
        self.preprocessed_trajs = None
        
        
        # Here we register some of the callbacks that must be managed by this class in order to show
        # the results in the web interface. 
        self.app.callback \
        (
            Output(component_id = 'user_result-'  + self.id_class, component_property = 'children'),   
            Input(component_id = 'user_sel-' + self.id_class, component_property = 'value')
        )(self.info_stops)
        
        ### Here we define and register all the callbacks that must be managed by the instance of this class ###
        self.app.callback \
        (
            Output(component_id = 'loading-' + self.id_class + '-c', component_property='children'),
            Output(component_id = 'output-' + self.id_class, component_property='children'),
            State(component_id = self.id_class + '-path', component_property='value'),
            State(component_id = self.id_class + '-duration', component_property='value'),
            State(component_id = self.id_class + '-radius', component_property='value'),
            Input(component_id = self.id_class + '-run', component_property='n_clicks')
        )(self.get_input_and_execute)
        
        
        
    ### CLASS PUBLIC METHODS ###
    
    def register_prev_module(self, prev_module : InteractiveModuleInterface) :
        
        print(f"Registering prev module {prev_module} in module {self.id_class}")
        self.prev_module = prev_module  
        
    
    def populate_input_area(self) :
        
        web_components = []
        
        # Here we manage the case where no data is available from the previous module.
        if(self.prev_module.get_results() is None) :
            web_components.append(html.Span(children = f"No preprocessed trajectories available from the {self.prev_module.id_class} module!"))
            web_components.append(html.Br())
            web_components.append(html.Span(children = f"Please, provide a file containing them: "))
            web_components.append(dcc.Input(id = self.id_class + '-path',
                                            value = './data/temp_dataset/traj_cleaned.parquet',
                                            type = 'text',
                                            placeholder = 'Path to file...'))
            web_components.append(html.Br())
            web_components.append(html.Br())
            
        else : 
            web_components.append(dcc.Input(id = self.id_class + '-path',
                                            value = None,
                                            type = 'text',
                                            placeholder = 'Path to file...',
                                            style={'display':'none'}))

        web_components.append(html.Span(children = "Minimum duration of a stop (in minutes): "))
        web_components.append(dcc.Input(id = self.id_class + '-duration',
                                        value = 10,
                                        type = 'number',
                                        placeholder = 'minutes'))
        web_components.append(html.Br())
        web_components.append(html.Br())
        
        web_components.append(html.Span(children = "Spatial radius (in km) of the stop: "))
        web_components.append(dcc.Input(id = self.id_class + '-radius',
                                        value = 1,
                                        type = 'number',
                                        placeholder = 'Spatial radius'))
        web_components.append(html.Br())
        web_components.append(html.Br())
        
        web_components.append(html.Button(id = self.id_class + '-run', children='RUN'))           
        
        return web_components
    
    
    def get_input_and_execute(self, path, duration, radius, button_state):
        
        print(f"Eseguo get_input_and_execute del modulo {self.id_class}! Button state: {button_state}")
    
        outputs = []
        if button_state is not None :
        
            print(f"Eseguo if in get_input_and_execute del modulo {self.id_class}! {button_state}")
            
            # Check input.
            if (duration is None) or (radius is None):
                outputs.append(html.H6(children='Error: some input values were not provided!'))
                return None, outputs
                
            if path is not None :
                try :
                    self.preprocessed_trajs = pd.read_parquet(path)
                except BaseException :
                    outputs.append(html.H5("No file with the preprocessed trajectories found! Please, provide one!"))
                    return None, outputs
            else : self.preprocessed_trajs = self.prev_module.get_results()
                

            # Salva nei campi dell'istanza l'input passato 
            self.duration = duration
            self.radius = radius
            self.stops = None
            self.moves = None
            
            # Esegui il codice core dell'istanza.
            self.core()
            
            
            # Manage the output to show in the web interface.
            options = [{'label': i, 'value': i} for i in self.get_users()]
            outputs.append(html.Div(id = 'users' + self.id_class,
                                    children=[html.P(children='User selection:'),
                                              dcc.Dropdown(id='user_sel-' + self.id_class,
                                                           options = options,
                                                           style={'color':'#333'}),
                                              html.Div(id = 'user_result-'  + self.id_class)]))
            
            # Save the stops and moves that have been detected to disk.
            self.save_output()

        
        return None, outputs


    def core(self):       
        
        tdf = skmob.TrajDataFrame(self.preprocessed_trajs)

        ### stop detection ###
        stdf = detection.stops(tdf,
                               stop_radius_factor = 0.5, 
                               minutes_for_a_stop = self.duration, 
                               spatial_radius_km = self.radius, 
                               leaving_time = True)
        self.stops = stdf


        ### move detection ###
        trajs = tdf.copy()
        starts = stdf.copy()
        ends = stdf.copy()

        trajs.set_index(['tid','datetime'], inplace = True)
        starts.set_index(['tid','datetime'], inplace = True)
        ends.set_index(['tid','leaving_datetime'], inplace = True)

        traj_ids = trajs.index
        start_ids = starts.index
        end_ids = ends.index

        # some datetime into stdf are approximated. In order to retrieve moves, we have to check the exact datime into 
        # trajectory dataframe. We use `isin()` method to reduce time computation
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

        if starts_ != []:
            start_idx = start_idx + starts_
    
        if ends_ != []:
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
        self.moves = moves.copy()

        del end_df, start_df, traj_df
        
        
    def get_results(self) :
    
        if (self.stops is not None) and (self.moves is not None) :
            return self.stops, self.moves
        else : 
            return None
            
            
    def reset_state(self) :
    
        print(f"Resetting state of the module {self.id_class}")
        self.stops = None
        self.moves = None
        
        
    def info_stops(self, user):

        outputs = []

        if user is None: return outputs

        outputs.append(html.Br())
        num_trajs = self.get_trajectories(user)
        outputs.append(html.P(children='Number of trajectories found for this user: {}'.format(num_trajs)))
        num_stops = self.get_stops(user)
        outputs.append(html.P(children='Number of stops found for this user: {}'.format(num_stops)))
        mean_duration = self.get_duration(user)
        outputs.append(html.P(children='Stop average duration: {} minutes'.format(mean_duration)))

        return outputs


    def get_users(self):
        
        return self.moves['uid'].unique()


    def get_trajectories(self, uid):

        return len(self.moves[self.moves['uid']==uid]['tid'].unique())


    def get_stops(self, uid):

        return len(self.stops[self.stops['uid']==uid])


    def get_duration(self, uid):

        s = self.stops[self.stops['uid']==uid]
        s['duration'] = (s['leaving_datetime'] - s['datetime']).astype('timedelta64[m]')

        return round(s['duration'].mean(),2)
        
        
    def save_output(self) :
        
        self.stops.to_parquet('./data/temp_dataset/stops.parquet')
        self.moves.to_parquet('./data/temp_dataset/moves.parquet')