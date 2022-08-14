from jupyter_dash import JupyterDash
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
import plotly.express as px
from dash.dependencies import Input, Output, State, MATCH, ALL
import pandas as pd
import geopandas as gpd
import numpy as np
import json

from core.backend_new import *


### GLOBAL VARIABLES ###

# Instantiate Dash application.
app = JupyterDash(__name__)

# By default, Dash applies validation to your callbacks, which performs checks such as validating the types of callback arguments and checking to see whether the specified Input and Output components actually have the specified properties. For full validation, all components within your callback must exist in the layout when your app starts, and you will see an error if they do not.
# However, in the case of more complex Dash apps that involve dynamic modification of the layout (such as multi-page apps), not every component appearing in your callbacks will be included in the initial layout. You can remove this restriction by disabling callback validation like this:
app.config.suppress_callback_exceptions = True


# Object representing the pipeline to be executed.
pipeline = Pipeline(app)


# NOTA: 'ALL' e 'MATCH' fanno parte delle funzionalita' di pattern-matching associati alle callback.
# @app.callback\
# (
    # Output(component_id='display', component_property='children'),
    # Input(component_id='tabs-inline', component_property='value'),
    # Input(component_id={'type':'radio_items', 'index': ALL}, component_property='value')
# )
    


# @app.callback\
# (
    # Output(component_id='loading-output', component_property='children'),
    # Output(component_id='outputs', component_property='children'), 
    # Output(component_id='users', component_property='style'),  
    # Output(component_id='user_list',component_property='options'),
    # Output(component_id='users_', component_property='style'),
    # Output(component_id='user_list_',component_property='options'),
    # Output(component_id='0', component_property='disabled'),
    # Output(component_id='1', component_property='disabled'),
    # Output(component_id='2', component_property='disabled'),
    # Output(component_id='output_sm', component_property='style'),
    # State(component_id={'type':'radio_items','index':ALL}, component_property='value'),
    # State(component_id={'type':'input','index':ALL}, component_property='value'),
    # State(component_id='tabs-inline', component_property='value'),
    # Input(component_id='run', component_property='n_clicks')
# )
# def show_output(radio, inputs, tab, click):
    # ### --- TO DO --- ###
    # #
    # # try to use current triggered in order to raise error (state null value)

    # disable0 = False
    # disable1 = True
    # disable2 = True
    # outputs = []
    # options = []
    # options2 = []
    # display = {'display':'none'}
    # display2 = {'display':'none'}
    # display_sm = {'display':'none'}
    
    
    # if click is None:
        # if tab != 'Preprocessing' and tab != 'tab-1':
            # disable0 = True
        # return None,outputs,display,options,display2,options2,disable0,disable1,disable2,display_sm

    
    # # Here we check if all the inputs required by some specific module have been provided.
    # is_empty = False
    # for input in inputs:
        # if input is None:
            # is_empty = True
    # if is_empty:
        # return None,\
               # html.H5(children='Please, insert input values!',style={'color':'red'}),\
               # None,None,None,None,None,None,None,None



    # elif tab == 'Enrichment':
    
        # print("Clicked the RUN button for enrichment!")

        # name_class = radio[2]
        # global class_e
        # class_ = globals()[name_class]
        # class_e = class_(inputs)   
        # class_e.core()
        # users = class_e.get_users()
        # display_sm = {'display':'none'}
        # display = {'display':'none'}
        # display2 = {'display':'inline'}
        # options2=[{'label': i, 'value': i} for i in users]
        
        # disable0 = True
        # disable1 = True


    # return None,outputs,display,options,display2,options2,disable0,disable1,disable2,display_sm



# @app.callback\
# (
    # Output(component_id='trajs', component_property='style'),   
    # Output(component_id='trajs_list',component_property='options'),
    # Input(component_id='user_list_',component_property='value'),
# )
# def trajectories(user):

    # display = {'display':'none'}
    # options = []

    # if user is None:
        # return display, options
    
    # display = {'display':'inline'}
    # trajs = class_e.get_trajectories(user)
    # options=[{'label': i, 'value': i} for i in trajs]

    # return display,options


# @app.callback\
# (
    # Output(component_id='outputs3',component_property='children'),
    # Input(component_id='user_list_',component_property='value')
# )
# def info_trajs(users):

    # outputs = []

    # if users is None:
        # return None

    # num_systematic = class_e.get_systematic(users)
    # num_occasional = class_e.get_occasional(users)
    # duration_transport = class_e.get_transport_duration(users)
    # duration_walk = duration_transport[duration_transport['label']==0]['datetime'].astype(str).values
    # duration_bike = duration_transport[duration_transport['label']==1]['datetime'].astype(str).values
    # duration_bus = duration_transport[duration_transport['label']==2]['datetime'].astype(str).values
    # duration_car = duration_transport[duration_transport['label']==3]['datetime'].astype(str).values
    # duration_subway = duration_transport[duration_transport['label']==4]['datetime'].astype(str).values
    # duration_train = duration_transport[duration_transport['label']==5]['datetime'].astype(str).values
    # duration_taxi = duration_transport[duration_transport['label']==6]['datetime'].astype(str).values

    # if len(duration_walk) == 0:
        # duration_walk = 0
    # else:
        # duration_walk = duration_walk[0]

    # if len(duration_bike) == 0:
        # duration_bike = 0
    # else:
        # duration_bike = duration_bike[0]

    # if len(duration_bus) == 0:
        # duration_bus = 0
    # else:
        # duration_bus = duration_bus[0]

    # if len(duration_car) == 0:
        # duration_car = 0
    # else:
        # duration_car = duration_car[0]
    
    # if len(duration_subway) == 0:
        # duration_subway = 0
    # else:
        # duration_subway = duration_subway[0]
    
    # if len(duration_train) == 0:
        # duration_train = 0
    # else:
        # duration_train = duration_train[0]

    # if len(duration_taxi) == 0:
        # duration_taxi = 0
    # else:
        # duration_taxi = duration_taxi[0]

    # tweets = class_e.get_tweets(users)

    # outputs.append(html.H6(children='Stops info:',style={'font-weight':'bold'}))
    # outputs.append(html.Span(children='N. systematic stops:',style={'text-decoration':'underline'}))
    # outputs.append(html.Span(children=str(num_systematic)+' \t'))
    # outputs.append(html.Span(children='N. occasional stops:',style={'text-decoration':'underline'}))
    # outputs.append(html.Span(children=str(num_occasional)))
    
    # outputs.append(html.H6(children='Transport mean info (duration):',style={'font-weigth':'bold'}))
    # outputs.append(html.Span(children='Walk:',style={'text-decoration':'underline'}))    
    # outputs.append(html.Span(children=str(duration_walk)+' \t'))
    # outputs.append(html.Span(children='Bike:',style={'text-decoration':'underline'}))    
    # outputs.append(html.Span(children=str(duration_bike)+' \t'))
    # outputs.append(html.Span(children='Bus:',style={'text-decoration':'underline'}))    
    # outputs.append(html.Span(children=str(duration_bus)+' \t'))
    # outputs.append(html.Br())
    # outputs.append(html.Span(children='Car:',style={'text-decoration':'underline'}))    
    # outputs.append(html.Span(children=str(duration_car)+' \t'))
    # outputs.append(html.Span(children='Train:',style={'text-decoration':'underline'}))    
    # outputs.append(html.Span(children=str(duration_train)+' \t'))
    # outputs.append(html.Span(children='Subway:',style={'text-decoration':'underline'}))    
    # outputs.append(html.Span(children=str(duration_subway)+' \t'))
    # outputs.append(html.Span(children='Taxi:',style={'text-decoration':'underline'}))    
    # outputs.append(html.Span(children=str(duration_taxi)+' \t'))
    # outputs.append(html.Br())

    # if len(tweets) != 0:
        # outputs.append(html.H6(children='Tweets:',style={'font-weigth':'bold'}))
        # for t in tweets:
            # outputs.append(html.Span(children='Tweet text:',style={'text-decoration':'underline'}))
            # outputs.append(html.Span(children='\"'+str(t)+'\"'))
            # outputs.append(html.Br())

    # class_e.get_transport_duration(users)

    # return outputs


# @app.callback\
# (
    # Output('output-maps','children'),
    # State(component_id='user_list_',component_property='value'),
    # Input(component_id='trajs_list',component_property='value')
# )
# def info_enrichment(user, traj):

    # # Dictionary holding the transportation modes mapping. 
    # transport = { 
        # 0: 'walk',
        # 1: 'bike',
        # 2: 'bus',
        # 3: 'car',
        # 4: 'subway',
        # 5: 'train',
        # 6: 'taxi'}

    # if user is None or traj is None:
        # return None

    # # Get the dataframes of interest from the enrichment class.
    # mats_moves, mats_stops, mats_systematic = class_e.get_mats(user,traj)


    # ### Preparing the information concerning the moves ###

    # #print(mats_moves['label'].unique())
    # mats_moves['label'] = mats_moves['label'].map(transport)
    # fig = px.line_mapbox(mats_moves,
                         # lat="lat",
                         # lon="lng",
                         # color="tid",
                         # hover_data=["label","temperature","w_conditions"],
                         # labels={"label":"transportation mean", "w_conditions":"weather condition"})
    
    
    
    # ### Prepare the information concerning the occasional stops... ###
    
    # mats_stops.drop_duplicates(subset=['category','distance'], inplace = True)
    
    # matched_pois = []
    # if ~mats_stops['distance'].isna().any() :
        # mats_stops['distance'] = round(mats_stops['distance'],2).astype(str)
        # mats_stops['description'] = '</br><b>PoI category</b>: ' +\
                                    # mats_stops['category'] +\
                                    # ' <b>Distance</b>: ' +\
                                    # mats_stops['distance']
        
        # limit_pois = 10
        # gb_occ_stops = mats_stops.groupby('stop_id')
        # for key, item in gb_occ_stops:
        
            # tmp = item['description']
            # size = tmp.shape[0]
            # limit = min(size, limit_pois)
            
            # tmp = tmp.head(limit)
            # stringa = tmp.str.cat(sep = "")
            # if size > limit_pois : 
                # stringa = stringa + f"</br>(...and other {size - limit_pois} POIs)"
                
            # matched_pois.append(stringa)
            
    # else :
        # matched_pois.append("No POI could be associated to this occasional stop!")


    # fig.add_trace(go.Scattermapbox(mode = "markers", name = 'occasional stops',
                                   # lon = mats_stops.lng.unique(),
                                   # lat = mats_stops.lat.unique(),
                                   # text = matched_pois,
                                   # hoverinfo = 'text',
                                   # marker = {'size': 10, 'color': '#F14C2B'}))



    # ### Preparing the information concerning the systematic stops ###

    # mats_systematic['home'] = round((mats_systematic['home']*100),2).astype(str)
    # mats_systematic['work'] = round((mats_systematic['work']*100),2).astype(str)
    # mats_systematic['other'] = round((mats_systematic['other']*100),2).astype(str)
    # mats_systematic['frequency'] = mats_systematic['frequency'].astype(str)

    # mats_systematic['description'] = '<b> Home </b>: ' + mats_systematic['home'] + '% </br></br> <b> Work </b>: ' + mats_systematic['work'] + '% </br> <b> Other </b>: ' + mats_systematic['other'] + '% </br> <b> Frequency </b>: ' + mats_systematic['frequency']
    # systematic_desc = list(mats_systematic['description'])

    # fig.add_trace(go.Scattermapbox(mode = "markers", name = 'systematic stops',
                                   # lon = mats_systematic.lng,
                                   # lat = mats_systematic.lat,
                                   # text = systematic_desc,
                                   # hoverinfo = 'text',
                                   # marker = {'size': 10,'color': '#2BD98C'}))
    
    
    
    # ### Setting the last parameters... ###
    
    # fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=12,
                      # margin={"r":0,"t":0,"l":0,"b":0})
    # fig.update_traces(line=dict(color='#2B37F1',width=2))

    # return dcc.Graph(figure=fig)



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


    ### Here we set up the tabs, which depend on the subclasses found in demo. ###
    ### These tabs are added to the list children_tabs, which will be then inserted into the web interface. ###
    children_tabs = []
    first_module = True
    for id, instance in pipeline.get_modules().items() :
        
        print(f"Module found: {id} -- {instance}")
        
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
                                         disabled = True, 
                                         disabled_style = disabled_style))



    ### Here we set up the individual components of the web interface ###
    
    title = html.Div(id='title',
                     children = [
                        html.Img(src='assets/MAT-Builder-logo.png',
                                 style={'width':'25%','height':'5%','float':'left'}),
                        html.Img(src='assets/loghi_mobidatalab.png',
                                 style={'width':'35%','height':'15%','float':'right'})],
                     style={'display':'inline-block','background-color':'white','padding':'1%'})
         
         
    input_area = html.Div(id='inputs',
                          children=[dcc.Tabs(id="tabs-inline", 
                                             children=children_tabs, 
                                             style=tabs_styles,
                                             value='None'),
                                    html.Br(),
                                    html.Div(id='display')],
                          style={'float':'left','width':'40%'})
                          
                          
    output_area = html.Div(style={'float':'right','width':'50%'},
                           children=[html.Div(id='outputs'),            
                                     html.Div(id='users_',
                                              children=[html.P(children='Users:'),
                                              dcc.Dropdown(id='user_list_',style={'color':'#333'})],
                                              style={'display':'none'}),
                                     html.Br(),
                                     html.Br(),
                                     html.Div(id='outputs3'),
                                     html.Br(),
                                     html.Br(),
                                     html.Div(id='trajs',
                                              children=[html.P(children='Trajectories:'), 
                                                        dcc.Dropdown(id='trajs_list',
                                                        style={'color':'#333'})],
                                              style={'display':'none'}),
                                     html.Br(),
                                     html.Div(id='output-maps')])
    
    
    
    # ### Here we arrange the layout of the individual components within the overall web interface ###
    app.layout = html.Div([title,
                           html.Br(),
                           html.Br(),
                           input_area,
                           output_area])

    app.run_server(debug=True)



if __name__ == '__main__':
    main()