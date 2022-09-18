import geopandas as gpd
import pandas as pd

import skmob
from skmob.preprocessing import filtering, compression
from ptrail.core.TrajectoryDF import PTRAILDataFrame

from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL

import plotly.express as px

from core.ModuleInterface import ModuleInterface



class Preprocessing(ModuleInterface):
    '''
    `preprocessing1` is a subclass of `Preprocessing` to preprocess trajectories and allows users to:
    1) remove trajectories with a few number of points
    2) remove outliers
    3) compress trajectories
    '''


    ### STATIC FIELDS ###
    
    id_class = 'Preprocessing'
    

    ### CLASS CONSTRUCTOR ###
    
    def __init__(self, app, pipeline):

        ### Here we register the Dash application ###
        self.app = app
        self.pipeline = pipeline
        self.prev_module = None
        
        self.path_output = './data/temp_dataset/traj_cleaned.parquet'
        self.df = None

        ### Here we define and register all the callbacks that must be managed by the instance of this class ###
        self.app.callback \
        (
            Output(component_id = 'loading-' + self.id_class + '-c', component_property='children'),
            Output(component_id = 'output-' + self.id_class, component_property='children'),
            State(component_id = self.id_class + '-path', component_property='value'),
            State(component_id = self.id_class + '-speed', component_property='value'),
            State(component_id = self.id_class + '-n_points', component_property='value'),
            Input(component_id = self.id_class + '-run', component_property='n_clicks')
        )(self.get_input_and_execute)



    ### CLASS METHODS ###
    
    def register_prev_module(self, prev_module) :
        
        print(f"Registering prev module {prev_module} in module {self.id_class}")
        self.prev_module = prev_module     
    
    
    def populate_input_area(self) :
        
        web_components = []
        
        web_components.append(html.Span(children = "Path to the raw trajectory dataset: "))
        web_components.append(dcc.Input(id = self.id_class + '-path',
                                        value = './data/Rome/rome.parquet',
                                        type = 'text',
                                        placeholder = 'path'))
        web_components.append(html.Br())
        web_components.append(html.Br())
        
        web_components.append(html.Span(children = "Outlier detection value (in km/h): "))
        web_components.append(dcc.Input(id = self.id_class + '-speed',
                                        value = 300,
                                        type = 'number',
                                        placeholder = 300))
        web_components.append(html.Br())
        web_components.append(html.Br())
        
        web_components.append(html.Span(children = "Minimum number of samples: "))
        web_components.append(dcc.Input(id = self.id_class + '-n_points',
                                        value = 3000,
                                        type = 'number',
                                        placeholder = 3000))
        web_components.append(html.Br())
        web_components.append(html.Br())
        
        web_components.append(html.Button(id = self.id_class + '-run', children='RUN'))           
        
        return web_components
        
        
    def get_input_and_execute(self, path, speed, n_points, button_state) :
    
        print(f"Eseguo get_input_and_execute del modulo {self.id_class}! Button state: {button_state}")
    
        outputs = []
        next_module_disabled = True
        if button_state is not None :
        
            print(f"Eseguo if in get_input_and_execute del modulo {self.id_class}! {button_state}")
            
            # Check input.
            if (speed is None) or (n_points is None):
                outputs.append(html.H6(children='Error: some input values were not provided!'))
                return None, outputs


            # Salva nei campi dell'istanza l'input passato 
            self.path = path
            self.kmh = speed
            self.num_point = n_points
            
            # Esegui il codice core dell'istanza.
            results = self.core()
            
            # Manage the output to show in the web interface.
            if results != '':
                outputs.append(html.H6(children='File not found or not valid path'))
                
            else:
                outputs.append(html.H6(children='Dataset statistics'))
                outputs.append(html.Hr())
                outputs.append(html.P(children='Tot. users: {} \t\t\t Tot. trajectories: {}'.format(self.get_num_users(), self.get_num_trajs())))
                outputs.append(dcc.Graph(figure = px.histogram(self.df.groupby('tid').datetime.first(),
                                         x = 'datetime',
                                         title = 'Distribution of trajectories over time')))
                
                # Save the results of the preprocessing to a file.
                self.output()
        
        
        return None, outputs
        
    
    def core(self):
                
        if self.path[-3:] == 'csv':
            gdf = pd.read_csv(self.path)
        elif self.path[-7:] == 'parquet':
            gdf = pd.read_parquet(self.path)
        else:
            return 'Fail in uploading file'

        # ## PREPROCESSING

        # eliminate trajectories with a number of points lower than num_point
        grouped = gdf.groupby('traj_id')
        gdf = grouped.filter(lambda x: len(x) >= self.num_point)

        # convert GeoDataFrame into pandas DataFrame
        df = pd.DataFrame(gdf)
        
        # now create a TrajDataFrame from the pandas DataFrame
        tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', datetime='time', user_id='user',
                                  trajectory_id='traj_id')
        ftdf = filtering.filter(tdf, max_speed_kmh = self.kmh)
        ctdf = compression.compress(ftdf, spatial_radius_km = 0.2)

        self.df = ctdf
        return ''
        
        
    def get_results(self) :
        return self.df
        
        
    def reset_state(self) :
        
        print(f"Resetting state of the module {self.id_class}")
        self.df = None


    def get_num_users(self):
        return str(len(self.df.uid.unique()))


    def get_num_trajs(self):
        return str(len(self.df.tid.unique()))


    def output(self):
        self.df.to_parquet(self.path_output)