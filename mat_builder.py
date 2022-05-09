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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = JupyterDash(__name__,external_stylesheets=external_stylesheets)

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

modules = [cls for cls in demo.__subclasses__()]

children_tabs = []

index = 0

for module in modules:
    
    methods = [ {'label':cls.__name__,'value':cls.__name__} for cls in module.__subclasses__()]
    children_tabs.append(dcc.Tab(id=str(index),label=module.__name__,value=module.__name__,style=tab_style,selected_style=tab_selected_style,
                                 children=[dcc.RadioItems(id={'type':'radio_items','index':index}, options=methods),html.Button(id={'type':'button','index':index}, children='Submit')]))

    index += 1

app.layout = html.Div([
    
    html.H1(children='MAT-Builder'), 
    
    html.Div(id='inputs',children=[
        dcc.Tabs(id="tabs-inline", children=children_tabs, style=tabs_styles),
        html.Hr(),
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
        html.Hr(),
        html.Div(id='outputs2'),
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
        
                if elem == 'Input':

                    inputs.append(dcc.Input(id={'type':'input','index':c},type=p[elem]['type'],placeholder=p[elem]['placeholder']))
                    inputs.append(html.Hr())

                elif elem == 'Checklist':

                    inputs.append(dcc.Checklist(id={'type':'input','index':c},options=p[elem]['options']))

            c +=1

        inputs.append(html.Button(id='run',children='RUN'))
    
    return inputs
    
@app.callback(
    Output(component_id='loading-output', component_property='children'),
    Output(component_id='outputs', component_property='children'), 
    Output(component_id='users', component_property='style'),  
    Output(component_id='user_list',component_property='options'),
    State(component_id={'type':'radio_items','index':ALL}, component_property='value'),
    State(component_id={'type':'input','index':ALL}, component_property='value'),
    State(component_id='tabs-inline', component_property='value'),
    Input(component_id='run', component_property='n_clicks')
)

def show_output(radio,inputs,tab,click):
    ### --- TO DO --- ###
    #
    # try to use current triggered in order to raise error (state null value)

    #print(radio, inputs, tab)
    outputs = []
    options = []
    display = {'display':'none'}

    #print(dash.callback_context.triggered)
    #print()


    if click is None:
        return None,outputs,display,options

    is_empty = False

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

    elif tab == 'Segmentation':

        name_class = radio[1]
        global class_s
        class_ = globals()[name_class]
        class_s = class_(inputs)   
        class_s.core()
        users = class_s.get_users()

        display = {'display':'inline'}
        options=[{'label': i, 'value': i} for i in users]

    elif tab == 'Enrichment':

        name_class = radio[2]
        global class_e
        class_ = globals()[name_class]
        print(inputs)
        class_e = class_(inputs)   
        class_e.core()

    return None,outputs,display,options

@app.callback(
    Output(component_id='outputs2', component_property='children'),   
    Input(component_id='user_list',component_property='value')
)

def info_stops(user):

    outputs = []

    if user is None:

        return outputs

    num_trajs = class_s.get_trajectories(user)
    print(num_trajs)
    outputs.append(html.Div(children='N. trajectories: {}'.format(num_trajs)))

    return outputs


if __name__ == '__main__':
    app.run_server(debug=True)
