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
from core.backend import *
import json

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = JupyterDash(__name__)#external_stylesheets=external_stylesheets)

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #ddc738',
    'borderLeft': '0px',
    'borderRight': '0px',
    'borderTop': '0px',
    'padding': '6px',
    'fontWeight': 'bold',
    'backgroundColor': '#131313'
}

tab_selected_style = {
    'borderTop': '1px solid',
    'borderLeft': '1px solid',
    'borderRight': '1px solid',
    'borderBottom': '0px',
    'borderColor': '#ddc738',
    'color': '#ddc738',
    'backgroundColor': '#131313',
    'padding': '6px'
}

disabled_style = {
    'borderBottom': '1px solid #ddc738',
    'borderLeft': '0px',
    'borderRight': '0px',
    'borderTop': '0px',
    'padding': '6px',
    'fontWeight': 'bold',
    'backgroundColor': '#131313',
    'color': '#5d5d5d'
}

modules = [cls for cls in demo.__subclasses__()]

children_tabs = []

index = 0

for module in modules:
    
    methods = [ {'label':cls.__name__,'value':cls.__name__} for cls in module.__subclasses__()]
    if index == 0:
        children_tabs.append(dcc.Tab(id=str(index),label=module.__name__,value=module.__name__,style=tab_style,selected_style=tab_selected_style,disabled_style=disabled_style,
                                 children=[dcc.RadioItems(id={'type':'radio_items','index':index}, options=methods),html.Button(id={'type':'button','index':index}, children='Submit')]))
    else:
        children_tabs.append(dcc.Tab(id=str(index),label=module.__name__,value=module.__name__,style=tab_style,selected_style=tab_selected_style,disabled=True, disabled_style=disabled_style,
                                 children=[dcc.RadioItems(id={'type':'radio_items','index':index}, options=methods),html.Button(id={'type':'button','index':index}, children='Submit')]))
    index += 1

app.layout = html.Div([
    
    html.H1(children='MAT-Builder'), 
    
    html.Div(id='inputs',children=[
        dcc.Tabs(id="tabs-inline", children=children_tabs, style=tabs_styles),
        html.Br(),
        html.Br(),
        html.Div(id='display'),
        html.Button(id='run',children='RUN',style={'display':'none'})],
        style={'float':'left','width':'40%'}),

    html.Div(style={'float':'right','width':'50%'}, children=[
        dcc.Loading(id="loading-1",
                    children=[html.Div([html.Div(id="loading-output")])],
        type="circle"),
        html.Div(id='outputs'),
        html.Div(id='users',children=[html.P(children='Users:'),
            dcc.Dropdown(id='user_list')
        ],style={'display':'none'}),
        html.Br(),
        html.Br(),
        html.Div(id='outputs2'),
        html.Div(id='users_',children=[html.P(children='Users:'),
            dcc.Dropdown(id='user_list_')
        ],style={'display':'none'}),
        html.Br(),
        html.Br(),
        html.Div(id='outputs3'),
        html.Br(),
        html.Br(),
        html.Div(id='trajs',children=[html.P(children='Trajectories:'),
            dcc.Dropdown(id='trajs_list')
        ],style={'display':'none'}),
        html.Div(id='output-maps'),
    ]),
    
])

@app.callback(
    Output(component_id='display', component_property='children'),
    Input(component_id='tabs-inline', component_property='value'),
    Input(component_id={'type':'radio_items','index':ALL}, component_property='value')
)

def show_input(tab,radio):

    inputs = []
    method = ''

    if tab is None:
        return 

    current_state = dash.callback_context.triggered
    
    if current_state[0]['prop_id'] != 'tabs-inline.value':

        method = current_state[0]['value']

    f = open('config.json')

    data = json.load(f)

    if(method != ''):
        parameters = data[method]
        c = 0 
        for p in parameters:
            
            for elem in p:
                
                if elem == 'H5':

                    inputs.append(html.H5(children=p[elem]['children']))                    

                if elem == 'Span':

                    inputs.append(html.Span(children=p[elem]['children']))

                if elem == 'Input':

                    inputs.append(dcc.Input(id={'type':'input','index':c},type=p[elem]['type'],placeholder=p[elem]['placeholder']))
                    inputs.append(html.Br())
                    inputs.append(html.Br())

                elif elem == 'Checklist':

                    inputs.append(dcc.Checklist(id={'type':'input','index':c},options=p[elem]['options']))

                elif elem == 'Dropdown':
                    if p[elem]['id'] == 'list_poi':
                        print('hello')
                        inputs.append(dcc.Dropdown(id={'type':'input','index':c},options=p[elem]['options'],multi=True,style={'color':'#333'}))                    
                        inputs.append(html.Br())
                        inputs.append(html.Br())

                    else:
                        inputs.append(dcc.Dropdown(id={'type':'input','index':c},options=p[elem]['options'],style={'color':'#333'}))                    
                        inputs.append(html.Br())
                        inputs.append(html.Br())

            c +=1

        inputs.append(html.Button(id='run',children='RUN'))
    
    return inputs
    
@app.callback(
    Output(component_id='loading-output', component_property='children'),
    Output(component_id='outputs', component_property='children'), 
    Output(component_id='users', component_property='style'),  
    Output(component_id='user_list',component_property='options'),
    Output(component_id='users_', component_property='style'),
    Output(component_id='user_list_',component_property='options'),
    Output(component_id='0', component_property='disabled'),
    Output(component_id='1', component_property='disabled'),
    Output(component_id='2', component_property='disabled'),
    State(component_id={'type':'radio_items','index':ALL}, component_property='value'),
    State(component_id={'type':'input','index':ALL}, component_property='value'),
    State(component_id='tabs-inline', component_property='value'),
    Input(component_id='run', component_property='n_clicks')
)

def show_output(radio,inputs,tab,click):
    ### --- TO DO --- ###
    #
    # try to use current triggered in order to raise error (state null value)

    disable1 = True
    disable2 = True
    outputs = []
    options = []
    options2 = []
    display = {'display':'none'}
    display2 = {'display':'none'}

    if radio!=[]:

        disable0 = True

    if click is None:
        disable0 = False
        return None,outputs,display,options,display2,options2,disable0,disable1,disable2

    is_empty = False

    len(inputs)

    for input in inputs:

        if input is None:

            is_empty = True

    if is_empty:
        return None,html.H5(children='Please, insert input values!',style={'color':'red'})

    if tab == 'Preprocessing':

        name_class = radio[0]

        global class_pp
        class_ = globals()[name_class]
        class_pp = class_(inputs)    
        class_pp.core()

        outputs.append(html.H6(children='Dataset statistics'))
        outputs.append(html.Hr())
        outputs.append(html.P(children='Tot. users: {} \t\t\t Tot. trajectories: {}'.format(class_pp.get_num_users(), class_pp.get_num_trajs())))
        outputs.append(dcc.Graph(figure=px.histogram(class_pp.df.datetime,x='datetime')))

        class_pp.output()
        disable0 = True
        disable1 = False

    elif tab == 'Segmentation':

        name_class = radio[1]
        global class_s
        class_ = globals()[name_class]
        class_s = class_(inputs)   
        class_s.core()
        users = class_s.get_users()

        display = {'display':'inline'}
        options=[{'label': i, 'value': i} for i in users]

        disable0 = True
        disable2 = False

    elif tab == 'Enrichment':

        name_class = radio[2]
        global class_e
        class_ = globals()[name_class]
        #print(inputs)
        class_e = class_(inputs)   
        class_e.core()
        users = class_e.get_users()

        display2 = {'display':'inline'}
        options2=[{'label': i, 'value': i} for i in users]

        disable0 = True
        disable1 = True

    return None,outputs,display,options,display2,options2,disable0,disable1,disable2

@app.callback(
    Output(component_id='outputs2', component_property='children'),   
    Input(component_id='user_list',component_property='value')
)

def info_stops(user):

    outputs = []

    if user is None:

        return outputs

    num_trajs = class_s.get_trajectories(user)
    outputs.append(html.Div(children='N. trajectories: {}'.format(num_trajs)))
    num_stops = class_s.get_stops(user)
    outputs.append(html.Div(children='N. stops: {}'.format(num_stops)))
    mean_duration = class_s.get_duration(user)
    outputs.append(html.Div(children='Stop average duration: {} minutes'.format(mean_duration)))

    return outputs

@app.callback(
    Output(component_id='trajs', component_property='style'),   
    Output(component_id='trajs_list',component_property='options'),
    Input(component_id='user_list_',component_property='value'),
)

def trajectories(user):

    display = {'display':'none'}
    options = []

    if user is None:
        return display, options

    display = {'display':'inline'}
    trajs = class_e.get_trajectories(user)
    options=[{'label': i, 'value': i} for i in trajs]

    return display,options

@app.callback(
    Output(component_id='outputs3',component_property='children'),
    Input(component_id='user_list_',component_property='value')
)

def info_trajs(users):

    outputs = []

    if users is None:

        return None

    num_systematic = class_e.get_systematic(users)
    outputs.append(html.Div(children='N. systematic stops: {}'.format(num_systematic)))
    num_occasional = class_e.get_occasional(users)
    outputs.append(html.Div(children='N. occasional stops: {}'.format(num_occasional)))

    return outputs

@app.callback(
    Output('output-maps','children'),
    State(component_id='user_list_',component_property='value'),
    Input(component_id='trajs_list',component_property='value')
)

def info_enrichment(user,traj):

    outputs = []

    if user is None or traj is None:

        return None, None

    mats_moves, mats_stops, mats_systematic = class_e.get_mats(user,traj)

    fig = px.line_mapbox(mats_moves, lat="lat", lon="lng", color="tid", hover_name="label")
    
    mats_stops['distance'] = round(mats_stops['distance'],2).astype(str)
    mats_stops['description'] = mats_stops['category'] + ' ' + mats_stops['distance']

    matched_pois = list(mats_stops.groupby('stop_id')['description'].agg("</br>".join))

    fig.add_trace(go.Scattermapbox(mode = "markers", name = 'occasional stops',
                                   lon = mats_stops.lng.unique(),
                                   lat = mats_stops.lat.unique(),
                                   text = matched_pois,
                                   hoverinfo = 'text',
                                   marker = {'size': 10, 'color': '#FF7233'}))

    mats_systematic['home'] = round((mats_systematic['home']*100),2).astype(str)
    mats_systematic['work'] = round((mats_systematic['work']*100),2).astype(str)
    mats_systematic['other'] = round((mats_systematic['other']*100),2).astype(str)
    mats_systematic['frequency'] = mats_systematic['frequency'].astype(str)

    mats_systematic['description'] = '<b> Home </b>: ' + mats_systematic['home'] + '% </br> <b> Work </b>: ' + mats_systematic['work'] + '% </br> <b> Other </b>: ' + mats_systematic['other'] + '% </br> <b> Frequency </b>: ' + mats_systematic['frequency']
    systematic_desc = list(mats_systematic['description'])

    fig.add_trace(go.Scattermapbox(mode = "markers", name = 'systematic stops',
                                   lon = mats_systematic.lng,
                                   lat = mats_systematic.lat,
                                   text = systematic_desc,
                                   hoverinfo = 'text',
                                   marker = {'size': 10,'color': '#74C869'}))
    
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=12,
                      margin={"r":0,"t":0,"l":0,"b":0})

    fig.update_traces(line=dict(color='#3392FF',width=2))

    return dcc.Graph(figure=fig)   


if __name__ == '__main__':
    app.run_server(debug=True)
