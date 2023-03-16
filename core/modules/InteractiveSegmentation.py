import pandas as pd

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL

from core.InteractiveModuleInterface import InteractiveModuleInterface
from core.InteractivePipeline import InteractivePipeline
from .InteractivePreprocessing import InteractivePreprocessing
from .Segmentation import Segmentation


class InteractiveSegmentation(InteractiveModuleInterface):
    '''
    `InteractiveSegmentation` acts as a graphical wrapper of a `Segmentation` module instance.
    An instance of this class must be used within an instance of the 'InteractivePipeline' class.
    '''
    
    
    ### STATIC FIELDS ###
    
    id_class = 'Segmentation'



    ### PRIVATE METHODS ###

    def _info_segmentation(self, user):

        outputs = []

        if user is None: return outputs

        outputs.append(html.Br())
        num_trajs = self._get_trajectories(user)
        outputs.append(html.P(children='Number of trajectories found for this user: {}'.format(num_trajs)))

        num_moves = self._get_num_moves(user)
        outputs.append(html.P(children='Number of moves found for this user: {}'.format(num_moves)))
        if num_moves :
            avg_move_duration = self._get_avg_move_duration(user)
            outputs.append(html.P(children='Move average duration: {} minutes'.format(avg_move_duration)))

        num_stops = self._get_num_stops(user)
        outputs.append(html.P(children='Number of stops found for this user: {}'.format(num_stops)))
        if num_stops :
            avg_stop_duration = self._get_avg_stop_duration(user)
            outputs.append(html.P(children='Stop average duration: {} minutes'.format(avg_stop_duration)))


        return outputs

    def _get_users(self):

        return self.moves['uid'].unique()

    def _get_trajectories(self, uid):

        return len(self.moves[self.moves['uid'] == uid]['tid'].unique())

    def _get_num_stops(self, uid):

        return len(self.stops[self.stops['uid'] == uid])

    def _get_num_moves(self, uid):
        print(f"{self.moves}")
        return self.moves[self.moves['uid'] == uid]['move_id'].nunique()

    def _get_avg_stop_duration(self, uid):

        s = self.stops[self.stops['uid'] == uid].copy()
        s['duration'] = (s['leaving_datetime'] - s['datetime']).astype('timedelta64[m]')

        return round(s['duration'].mean(), 2)

    def _get_avg_move_duration(self, uid):

        gb = self.moves[self.moves['uid'] == uid].groupby('move_id')
        df = gb.agg({'datetime' : ['last', 'first']})
        df['duration'] = (df[('datetime','last')] - df[('datetime','first')]).astype('timedelta64[m]')
        return round(df['duration'].mean(), 2)

    def _save_output(self):

        self.stops.to_parquet('./data/temp_dataset/stops.parquet')
        self.moves.to_parquet('./data/temp_dataset/moves.parquet')


    ### CLASS CONSTRUCTOR ###
    
    def __init__(self, app : Dash, pipeline : InteractivePipeline) :

        self.prev_modules = {}

        self.app = app
        self.pipeline = pipeline
        self.segmentation : Segmentation = Segmentation()
        self.stops = None
        self.moves = None

        
        # Here we register some of the callbacks that must be managed by this class in order to show
        # the results in the web interface. 
        self.app.callback \
        (
            Output(component_id = 'user_result-'  + self.id_class, component_property = 'children'),   
            Input(component_id = 'user_sel-' + self.id_class, component_property = 'value')
        )(self._info_segmentation)


        ### Here we define and register all the callbacks that must be managed by the instance of this class ###
        self.app.callback \
        (
            Output(component_id = 'loading-' + self.id_class + '-c', component_property='children'),
            Output(component_id = 'output-' + self.id_class, component_property='children'),
            State(component_id = self.id_class + '-path', component_property='value'),
            State(component_id = self.id_class + '-duration', component_property='value'),
            State(component_id = self.id_class + '-radius', component_property='value'),
            Input(component_id = self.id_class + '-run', component_property='n_clicks')
        )(self.get_input_and_execute_task)

        
        
    ### CLASS PUBLIC METHODS ###

    def get_dependencies(self) -> list[InteractiveModuleInterface]:
        return [InteractivePreprocessing]
    
    def register_modules(self, list_modules: list[InteractiveModuleInterface]):

        print(f"Registering prev modules {list_modules} in module {self.id_class}")
        for m in list_modules :
            if type(m) == InteractivePreprocessing :
                print("Registering preprocessing module!")
                self.prev_modules[InteractivePreprocessing] = m
        
    
    def populate_input_area(self) :
        
        web_components = []
        
        # Here we manage the case where no data is available from the previous module.
        if(InteractivePreprocessing not in self.prev_modules) or\
          (next(iter(self.prev_modules[InteractivePreprocessing].get_results().values())) is None) :
            web_components.append(html.Span(children = f"No trajectory dataset available!"))
            web_components.append(html.Br())
            web_components.append(html.Span(children=f"Please, provide a path to one: "))
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
                                        value = 120,
                                        type = 'number',
                                        placeholder = 'minutes'))
        web_components.append(html.Br())
        web_components.append(html.Br())
        
        web_components.append(html.Span(children = "Maximum spatial radius (in meters) of a stop: "))
        web_components.append(dcc.Input(id = self.id_class + '-radius',
                                        value = 200,
                                        type = 'number',
                                        placeholder = 'Spatial radius'))
        web_components.append(html.Br())
        web_components.append(html.Br())
        
        web_components.append(html.Button(id = self.id_class + '-run', children='RUN'))           
        
        return web_components
    
    
    def get_input_and_execute_task(self, path, duration, radius, button_state):
        
        print(f"Eseguo get_input_and_execute del modulo {self.id_class}! Button state: {button_state}")
    
        outputs = []
        if button_state is not None :
        
            print(f"Eseguo if in get_input_and_execute del modulo {self.id_class}! {button_state}")
            
            # Check input.
            if [x for x in (duration, radius) if x is None] :
                outputs.append(html.H6(children='Error: some input values were not provided!'))
                return None, outputs

            preprocessed_trajs = None
            if path is not None :
                try :
                    preprocessed_trajs = pd.read_parquet(path)
                except BaseException :
                    outputs.append(html.H5("No file with the preprocessed trajectories found! Please, provide one!"))
                    return None, outputs
            else : preprocessed_trajs = self.prev_modules[InteractivePreprocessing].get_results()['preprocessed_trajectories']

            
            # Execute the segmentation module and retrieve the output as well as the execution .
            dic_params = {'trajectories' : preprocessed_trajs,
                          'duration' : duration,
                          'radius' : radius / 1000}
            is_exe_ok = self.segmentation.execute(dic_params)
            results = self.segmentation.get_results()
            self.stops = results['stops']
            self.moves = results['moves']

            
            # Manage the output to show in the web interface.
            outputs.append(html.H6(children=f"Overall number of moves found: {self.moves['move_id'].nunique()}",
                                   style={'font-weight': 'bold'}))
            outputs.append(html.H6(children=f'Overall number of stops found: {len(self.stops)}',
                                   style={'font-weight': 'bold'}))

            options = [{'label': i, 'value': i} for i in self._get_users()]
            outputs.append(html.Div(id = 'users' + self.id_class,
                                    children=[html.H6(children='Select a user for more specific information:',
                                                      style={'font-weight': 'bold'}),
                                              dcc.Dropdown(id='user_sel-' + self.id_class,
                                                           options = options,
                                                           style = {'color':'#333'}),
                                              html.Div(id = 'user_result-'  + self.id_class)]))
            
            # Save the stops and moves that have been detected to disk.
            self._save_output()

        
        return None, outputs

    def get_results(self) -> dict :
        return self.segmentation.get_results()

    def reset_state(self) :
    
        print(f"Resetting state of the module {self.id_class}")
        self.segmentation.reset_state()
        self.stops = None
        self.moves = None