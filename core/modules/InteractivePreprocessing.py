import pandas as pd
import os

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL

import plotly.express as px

from core.InteractiveModuleInterface import InteractiveModuleInterface
from .Preprocessing import Preprocessing


class InteractivePreprocessing(InteractiveModuleInterface):
    '''
    `InteractivePreprocessing` acts as a graphical wrapper of a `Preprocessing` instance. To be used within an
    'InteractivePipeline' object.
    '''


    ### STATIC FIELDS ###
    
    id_class = 'Preprocessing'



    ### PROTECTED CLASS METHODS ###

    def _get_num_users(self, df):
        return str(len(df.uid.unique()))

    def _get_num_trajs(self, df):
        return str(len(df.tid.unique()))



    ### PUBLIC CLASS CONSTRUCTOR ###
    
    def __init__(self, app : Dash, pipeline : list[InteractiveModuleInterface]):

        ### Here we register the Dash application ###
        self.preprocessing : Preprocessing = Preprocessing()
        self.app = app
        self.pipeline = pipeline
        self.prev_module = None
        self.path_output = './data/temp_dataset/traj_cleaned.parquet'

        ### Here we define and register all the callbacks that must be managed by the instance of this class ###
        self.app.callback \
        (
            Output(component_id = 'loading-' + self.id_class + '-c', component_property='children'),
            Output(component_id = 'output-' + self.id_class, component_property='children'),
            State(component_id = self.id_class + '-path', component_property='value'),
            State(component_id = self.id_class + '-speed', component_property='value'),
            State(component_id = self.id_class + '-n_points', component_property='value'),
            State(component_id = self.id_class + '-compress', component_property='value'),
            Input(component_id = self.id_class + '-run', component_property='n_clicks')
        )(self.get_input_and_execute)



    ### PUBLIC CLASS METHODS ###
    
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
        
        web_components.append(html.Span(children = "Minimum number of samples a trajectory must have: "))
        web_components.append(dcc.Input(id = self.id_class + '-n_points',
                                        value = 3000,
                                        type = 'number',
                                        placeholder = 3000))
        web_components.append(html.Br())
        web_components.append(html.Br())

        web_components.append(html.Span(children="Compress trajectories (this can speed up other modules): "))
        web_components.append(dcc.Dropdown(id=self.id_class + '-compress',
                                           options=[{"label": "yes", "value": "yes"},
                                                    {"label": "no", "value": "no"}],
                                           value="yes",
                                           style={'color': '#333'}))
        web_components.append(html.Br())
        web_components.append(html.Br())
        
        web_components.append(html.Button(id = self.id_class + '-run', children='RUN'))           
        
        return web_components
        
    def get_input_and_execute(self, path, speed, n_points, compress, button_state) :
    
        print(f"Eseguo get_input_and_execute del modulo {self.id_class}! Button state: {button_state}")
    
        outputs = []
        next_module_disabled = True
        if button_state is not None :
        
            print(f"Eseguo if in get_input_and_execute del modulo {self.id_class}! {button_state}")

            if (path is None) or (os.path.isfile(path) is False) :
                outputs.append(html.H6(children='Error: invalid path to the trajectory file!'))
                return None, outputs
            
            # Check input.
            if [x for x in (speed, n_points, compress) if x is None]:
                outputs.append(html.H6(children='Error: some input values were not provided!'))
                return None, outputs


            # Salva nei campi dell'istanza l'input passato

            dic_params = {'trajectories' : pd.read_parquet(path),
                          'speed' : speed,
                          'n_points' : n_points,
                          'compress' : True if 'yes' else False}
            # Esegui il codice core dell'istanza.
            self.preprocessing.execute(dic_params)
            results = self.preprocessing.get_results()['preprocessed_trajectories']
            
            # Manage the output to show in the web interface.
            outputs.append(html.H6(children='Dataset statistics'))
            outputs.append(html.Hr())
            outputs.append(html.P(children='Tot. users: {} \t\t\t Tot. trajectories: {}'.format(self._get_num_users(results), self._get_num_trajs(results))))
            outputs.append(dcc.Graph(figure = px.histogram(results.groupby('tid').datetime.first(),
                                     x = 'datetime',
                                     title = 'Distribution of trajectories over time')))

            # Save the results of the preprocessing to a file.
            results.to_parquet(self.path_output)
        
        return None, outputs

    def get_results(self) -> dict :
        return self.preprocessing.get_results()
        
    def reset_state(self) :
        
        print(f"Resetting state of the module {self.id_class}")
        self.preprocessing.reset_state()