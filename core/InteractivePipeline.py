from collections import OrderedDict
from . import InteractiveModuleInterface

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL



class InteractivePipeline():
    '''
    The class `InteractivePipeline` models an interactive pipeline that must manage a sequence of modules making up a semantic enrichment process.
    The pipeline executes the task logics of the modules sequentially -- for instance, a semantic enrichment process that includes the tasks
    `preprocessing`, `segmentation`, and `enrichment` will execute the task logic of the associated modules in that order.
    '''

    ### CLASS CONSTRUCTOR ###

    def __init__(self, app : Dash, list_modules : list[InteractiveModuleInterface]):
        '''
        The constructor of this class takes in input the type references of the modules that must manage, the order of execution,
        and the dependencies between their inputs and their outputs.

        Parameters
        ----------
        app : Dash
            Dash UI interface.
        list_modules : list[InteractiveModuleInterface]
            Type references of the interactive modules that must be executed by the interactive pipeline.
        '''
        
        ### Here we register the Dash app within the pipeline ###
        self.app = app
        
        # Define the modules of the pipeline and the order used to execute them.
        self.pipeline = OrderedDict()
        
        # Instantiate the modules to be used in the pipeline.
        for module in list_modules : self.pipeline[module.id_class] = module(app, self)
        
        
        # Make each module aware of the modules from which output it depends.
        print('Checking dependencies between modules...')
        for key, module in self.pipeline.items() :
            list_dependencies = module.get_dependencies()
            list_references = [x for x in self.pipeline.values() if type(x) in list_dependencies]
            module.register_modules(list_references)
          
          
        # Set up the CSS styles.
        self._setup_css_styles()
        
        # Set up the web interface layout.
        self._setup_app_layout()
        
        
        ### Here we register all the callbacks that must be managed by the pipeline instance ###
        self.app.callback \
        (
            Output(component_id = 'display', component_property='children'),
            Output(component_id = 'outputs', component_property='children'),
            Input(component_id = 'tabs-inline', component_property='value')
        )(self.setup_input_output_areas)
    
    
    
    ### PROTECTED METHODS ###
    
    def _setup_css_styles(self) :
        '''
        This method defines the CSS style to be used for the UI.
        '''
    
        # CSS style parameters declarations/definitions #

        # Here we define the styles to be applied to the tabs...
        self.tabs_styles = \
        {
            'height': '44px'
        }

        self.tab_style = \
        {
            'borderBottom': '1px solid #ddc738',
            'borderLeft': '0px',
            'borderRight': '0px',
            'borderTop': '0px',
            'padding': '6px',
            'fontWeight': 'bold',
            'backgroundColor': '#131313'
        }

        self.tab_selected_style = \
        {
            'borderTop': '1px solid',
            'borderLeft': '1px solid',
            'borderRight': '1px solid',
            'borderBottom': '0px',
            'borderColor': '#ddc738',
            'color': '#ddc738',
            'backgroundColor': '#131313',
            'padding': '6px'
        }

        self.disabled_style = \
        {
            'borderBottom': '1px solid #ddc738',
            'borderLeft': '0px',
            'borderRight': '0px',
            'borderTop': '0px',
            'padding': '6px',
            'fontWeight': 'bold',
            'backgroundColor': '#131313',
            'color': '#5d5d5d'
        }
    
    
    def _setup_app_layout(self) :
        '''
        This method sets up the initial state of the UI. The UI will be arranged according to the following areas:
            - Tab selector area: here the user will be able to select the module they want to execute.
            - Input area: area used by the modules to show to the users which input parameters are required, and to get from the
              users the values for such parameters.
            - Output area: area used by the modules to give graphical feedback once they have executed their internal task logic.
        '''
    
        ### Here we set up the tabs, which depend on the subclasses found in demo. ###
        ### These tabs are added to the list children_tabs, which will be then inserted into the web interface. ###
        children_tabs = []
        for id, instance in self.pipeline.items() :

            print(f"Creating tab for: {id} -- {instance}")
            children_tabs.append(dcc.Tab(id=id,
                                         label = id,
                                         value = id,
                                         style = self.tab_style,
                                         selected_style = self.tab_selected_style,
                                         disabled_style = self.disabled_style))



        ### Here we set up the individual components of the web interface ###
        title = html.Div(id='title',
                         children = [html.Img(src='assets/MAT-Builder-logo.png', 
                                              style={'width':'25%','height':'5%','float':'left'}),
                                     html.Img(src='assets/loghi_mobidatalab.png', 
                                              style={'width':'35%','height':'15%','float':'right'})],
                         style = {'display':'inline-block','background-color':'white','padding':'1%'})
             
             
        input_area = html.Div(id='inputs',
                              children=[dcc.Tabs(id="tabs-inline", 
                                                 children = children_tabs, 
                                                 style = self.tabs_styles,
                                                 value = 'None'),
                                        html.Br(),
                                        html.Div(id='display')],
                              style={'float':'left','width':'40%'})
                              
                              
        output_area = html.Div(style = {'float':'right','width':'50%'},
                               children = [html.Br(),
                                           html.Div(id='outputs')])
        
        
        
        # ### Here we arrange the layout of the individual components within the overall web interface ###
        self.app.layout = html.Div([title,
                                    input_area,
                                    output_area])
    
    
    
    ### PUBLIC METHODS ###
    
    def setup_input_output_areas(self, name_module : str):
        '''
        This method is invoked when a user selects a specific module in the tab selector area, and prepares the input and output areas
        specifically for the needs of the selected module.

        Parameters
        ----------
        name_module : str
            Identifier of the module that has been selected by the user in the tab selector area.
        '''

        print(f"show_input invoked! Tab: {name_module}")
        
        inputs = []
        output_area = []
        if name_module != 'None' :
        
            print(f"The module {self.pipeline[name_module].id_class} will populate the input area in the web interface...")
            inputs = self.pipeline[name_module].populate_input_area()
            
            print(f"The pipeline prepares the output area of the web interface for the module {self.pipeline[name_module].id_class}...")
            output_area = [dcc.Loading(id = "loading-" + self.pipeline[name_module].id_class,
                                       children = html.Div(html.Div(id = "loading-" + self.pipeline[name_module].id_class + '-c')), 
                                       type="default"),
                           html.Div(id = 'output-' + self.pipeline[name_module].id_class)]
        
        # Questo e' il caso in cui ci si trova all'avvio dell'app o refresh browser...si resetta lo stato dei moduli.
        else :
            for k, v in self.pipeline.items() :
                v.reset_state()
        
        
        return inputs, output_area