from dash import Dash, dcc, html
from core.backend_new import *



### MAIN application ###

def main() :

    # CSS style parameters declarations/definitions #

    # Here we define the styles to be applied to the tabs...
    tabs_styles = \
    {
        'height': '44px'
    }

    tab_style = \
    {
        'borderBottom': '1px solid #ddc738',
        'borderLeft': '0px',
        'borderRight': '0px',
        'borderTop': '0px',
        'padding': '6px',
        'fontWeight': 'bold',
        'backgroundColor': '#131313'
    }

    tab_selected_style = \
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

    disabled_style = \
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
    
    
    # Instantiate the Dash application.
    app = Dash(__name__)


    # By default, Dash applies validation to your callbacks, which performs checks such as validating the types of callback arguments and checking to see whether the specified Input and Output components actually have the specified properties. For full validation, all components within your callback must exist in the layout when your app starts, and you will see an error if they do not.
    # However, in the case of more complex Dash apps that involve dynamic modification of the layout (such as multi-page apps), not every component appearing in your callbacks will be included in the initial layout. You can remove this restriction by disabling callback validation like this:
    app.config.suppress_callback_exceptions = True


    # Object representing the pipeline to be executed.
    # TODO: separate the definition of the pipeline (i.e., the modules it must execute) from the definition of the class.
    #       We can do so by passing the list of modules here in the main.
    pipeline = Pipeline(app)


    ### Here we set up the tabs, which depend on the subclasses found in demo. ###
    ### These tabs are added to the list children_tabs, which will be then inserted into the web interface. ###
    children_tabs = []
    first_module = True
    for id, instance in pipeline.get_modules().items() :
        
        print(f"Creating tab for: {id} -- {instance}")
        
        if first_module :
            children_tabs.append(dcc.Tab(id = id,
                                         label = id,
                                         value = id,
                                         style = tab_style,
                                         selected_style = tab_selected_style,
                                         disabled_style = disabled_style))

            first_module = False                              
             
        else:
            children_tabs.append(dcc.Tab(id=id,
                                         label = id,
                                         value = id,
                                         style = tab_style,
                                         selected_style = tab_selected_style,
                                         disabled_style = disabled_style))



    ### Here we set up the individual components of the web interface ###
    # TODO: we can move this code in the Pipeline class.
    title = html.Div(id='title',
                     children = [html.Img(src='assets/MAT-Builder-logo.png', 
                                          style={'width':'25%','height':'5%','float':'left'}),
                                 html.Img(src='assets/loghi_mobidatalab.png', 
                                          style={'width':'35%','height':'15%','float':'right'})],
                     style = {'display':'inline-block','background-color':'white','padding':'1%'})
         
         
    input_area = html.Div(id='inputs',
                          children=[dcc.Tabs(id="tabs-inline", 
                                             children=children_tabs, 
                                             style=tabs_styles,
                                             value='None'),
                                    html.Br(),
                                    html.Div(id='display')],
                          style={'float':'left','width':'40%'})
                          
                          
    output_area = html.Div(style = {'float':'right','width':'50%'},
                           children = [html.Br(),
                                       html.Div(id='outputs')])
    
    
    
    # ### Here we arrange the layout of the individual components within the overall web interface ###
    app.layout = html.Div([title,
                           input_area,
                           output_area])


    app.run_server(debug=True)



if __name__ == '__main__':
    main()