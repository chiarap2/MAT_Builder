### SEMANTIC ENRICHMENT PROCESSOR API USE EXAMPLE ###
#
# Ensure that this script has access to the files of a dataset that can work with the processor.
# This example uses the Rome dataset, which is also present in the official GitHub repository (although it must be decompressed).
#
# Ensure also that this script is accessing the semantic enrichment processor's correct IP -- you can set it up
# in the "url_service" variable below.
#
# The script makes use of the 3 endpoints the semantic enrichment processor API exposes: preprocessing, segmentation, and enrichment.
# To this end, we refer to the "main" function at the end of the script, and to the various functions used by the main, each of which
# shows how to make POST and GET requests (i.e., which parameters and files must be passed) with the endpoints.


import time

import requests
import pandas as pd
import re
from typing import Optional, Tuple


### Semantic enrichment processor root address ###
# url_service = "http://127.0.0.1:8000/semantic/"
# url_service = "http://azureuser@semantic.westeurope.cloudapp.azure.com:8000/semantic/"
url_service = 'https://services.mobidatalab.eu:8443/semantic/'


def test_preprocessing_post(pathfile : str) -> Tuple[int, Optional[str]]:
    '''
    This function issues a POST request to the preprocessing endpoint of the semantic enrichment processor web API.
    '''

    # Here we set up the endpoint, parameters, and binary file to pass in the POST request.
    url = url_service + "Preprocessing/"
    parameters = {'max_speed' : 300, 'min_num_samples' : 1500, 'compress_trajectories' : True}
    files = {'file_trajectories': ('trajectories.parquet', open(pathfile, 'rb'))}
    res = requests.post(url, params=parameters, files=files)

    print(res)
    # print(res.json()['message'])

    # If the request was successful, return the task ID returned by the processor.
    # If an error occurred, then return None.
    if res.status_code == 200:
        return res.status_code, res.json()['message'].split(' ')[1]
    else:
        return res.status_code, None


def test_preprocessing_get(task_id : str) -> Tuple[int, Optional[str]]:
    '''
    This function issues a GET request to the preprocessing endpoint of the semantic enrichment processor web API.
    '''

    # Set up the API endpoint, parameters, and path where to save the file received from the server.
    url = url_service + "Preprocessing/"
    parameters = {'task_id' : task_id}
    res = requests.get(url, params=parameters)

    print(res)
    filename = None
    if res.status_code == 200 :
        filename = "preprocessed_trajectories.parquet"
        if "content-disposition" in res.headers.keys():
            filename = re.findall("filename=\"(.+)\"", res.headers['content-disposition'])[0]
        print(f"Writing received file to: {filename}")
        with open(filename, 'wb') as f:
            f.write(res.content)

    return res.status_code, filename

        #test_file = pd.read_parquet(filename)
        #print(test_file.info())


def test_segmentation_post(pathfile : str) -> Tuple[int, Optional[str]]:
    '''
    This function issues a POST request to the segmentation endpoint of the semantic enrichment processor web API.
    '''

    url = url_service + "Segmentation/"
    params = {'min_duration_stop': 10, 'max_stop_radius': 0.2}
    files = {'file_trajectories': ('trajectories.parquet', open(pathfile, 'rb'))}

    res = requests.post(url, params=params, files=files)

    print(res)
    # print(res.json()['message'])

    if res.status_code == 200:
        return res.status_code, res.json()['message'].split(' ')[1]
    else:
        return res.status_code, None


def test_segmentation_get(task_id : str) -> Tuple[int, Optional[str], Optional[str]]:
    '''
    This function issues a GET request to the segmentation endpoint of the semantic enrichment processor web API.
    '''

    url = url_service + "Segmentation/"
    parameters = {'task_id' : task_id}

    res = requests.get(url, params = parameters)
    print(res)
    print(res.status_code)

    stops_path = None
    moves_path = None
    if res.status_code == 200 :
        # Translate the received stops and moves from json to dataframes.
        stops = pd.DataFrame.from_dict(res.json()['stops'])
        moves = pd.DataFrame.from_dict(res.json()['moves'])

        print(stops.info())
        print(moves.info())

        # Store the dataframes into parquet files.
        stops_path = './stops.parquet'
        moves_path = './moves.parquet'
        stops.to_parquet(stops_path)
        moves.to_parquet(moves_path)

    return res.status_code, stops_path, moves_path


def test_enrichment_post(path_trajs : str,
                         path_moves : str,
                         path_stops : str,
                         path_pois : str,
                         path_social : str,
                         path_weather : str) -> Tuple[int, Optional[str]]:
    '''
    This function issues a POST request to the enrichment endpoint of the semantic enrichment processor web API.
    '''

    url = url_service + "Enrichment/"
    params = \
        {
            'move_enrichment': True,
            'max_dist': 50,
            'dbscan_epsilon': 50,
            'systematic_threshold': 5
        }
    files = \
        {
            'file_trajectories': ('trajectories.parquet', open(path_trajs, 'rb')),
            'file_moves': ('moves.parquet', open(path_moves, 'rb')),
            'file_stops': ('stops.parquet', open(path_stops, 'rb')),
            'file_pois': ('pois.parquet', open(path_pois, 'rb')),
            'file_social': ('social.parquet', open(path_social, 'rb')),
            'file_weather': ('weather.parquet', open(path_weather, 'rb'))
        }
    res = requests.post(url, params=params, files=files)

    print(res)
    print(res.json()['message'])

    if res.status_code == 200:
        return res.status_code, res.json()['message'].split(' ')[1]
    else:
        return res.status_code, None


def test_enrichment_get(task_id: str) -> Tuple[int, Optional[str]]:
    '''
    This function issues a GET request to the enrichment endpoint of the semantic enrichment processor web API.
    '''

    url = url_service + "Enrichment/"
    parameters = {'task_id': task_id}
    res = requests.get(url, params=parameters)

    print(res)
    filename = None
    if res.status_code == 200:
        filename = "results.ttl"
        if "content-disposition" in res.headers.keys():
            filename = re.findall("filename=\"(.+)\"", res.headers['content-disposition'])[0]
        print(f"Writing received file to: {filename}")
        with open(filename, 'wb') as f:
            f.write(res.content)

    return res.status_code, filename


def main() :
    '''
    In this main we simulate the chains of requests done to the semantic enrichment processor in order to,
    starting from a dataset of trajectories, get an RDF knowledge graph containing a dataset of semantically enriched
    trajectories.
    '''


    ### Step 1 - preprocessing the trajectory dataset. ###
    req_code, task_id = test_preprocessing_post('./datasets/rome/rome.parquet')
    if req_code == 200:
        print(f"Preprocessing POST request successful! Task ID: {task_id}")
    else:
        print(f"Preprocessing POST request not successful (code {req_code}), aborting...")
        return


    waiting = True
    path_preprocessed = None
    while waiting :
        req_code, path_preprocessed = test_preprocessing_get(task_id)

        if req_code == 200:
            print(f"Preprocessing GET request successful (task {task_id}, file received {path_preprocessed})!")
            waiting = False
        elif req_code == 404 :
            print(f"Server is still processing the preprocessing task {task_id} (code {req_code})")
            time.sleep(5)
        else:
            print(f"Server failed at processing the preprocessing task {task_id} (code {req_code}), aborting...")
            return



    ### Step 2 - segmenting the trajectories in the trajectory dataset. ###
    req_code, task_id = test_segmentation_post(path_preprocessed)
    if req_code == 200:
        print(f"Segmentation POST request successful! Task ID: {task_id}")
    else:
        print(f"Segmentation POST request not successful (code {req_code}), aborting...")
        return


    waiting = True
    stops_path = None
    moves_path = None
    while waiting :
        req_code, stops_path, moves_path = test_segmentation_get(task_id)

        if req_code == 200:
            print(f"Segmentation GET request successful (task {task_id}, files received: {stops_path}, {moves_path})!")
            waiting = False
        elif req_code == 404:
            print(f"Server is still processing the segmentation task {task_id} (code {req_code})")
            time.sleep(5)
        else:
            print(f"Server failed at processing the segmentation task {task_id} (code {req_code}), aborting...")
            return


    ### Step 3 - Trajectory enrichment. ###
    req_code, task_id = test_enrichment_post(path_preprocessed,
                                             moves_path,
                                             stops_path,
                                             './datasets/rome/poi/pois.parquet',
                                             './datasets/rome/tweets/tweets_rome.parquet',
                                             './datasets/rome/weather/weather_conditions.parquet')
    if req_code == 200:
        print(f"Enrichment POST request successful! Task ID: {task_id}")
    else:
        print(f"Enrichment POST request not successful (code {req_code}), aborting...")
        return


    waiting = True
    kg_path = None
    while waiting:
        req_code, kg_path = test_enrichment_get(task_id)

        if req_code == 200:
            print(f"Enrichment GET request successful (task {task_id}, KG file received: {kg_path})!")
            waiting = False
        elif req_code == 404:
            print(f"Server is still processing the enrichment task {task_id} (code {req_code})")
            time.sleep(5)
        else:
            print(f"Server failed at processing the enrichment task {task_id} (code {req_code}), aborting...")
            return


if __name__ == '__main__':
    main()