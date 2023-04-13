import requests
import pandas as pd
import re
from typing import Optional


# url_service = "http://127.0.0.1:8000/semantic/"
url_service = "http://azureuser@semantic.westeurope.cloudapp.azure.com:8000/semantic/"


def test_preprocessing_post(pathfile : str) -> Optional[str]:

    url = url_service + "Preprocessing/"
    parameters = {'max_speed' : 300, 'min_num_samples' : 1500, 'compress_trajectories' : True, 'token' : 'aaa'}
    files = {'file_trajectories': ('trajectories.parquet', open(pathfile, 'rb'))}
    res = requests.post(url, params=parameters, files=files)

    print(res)
    print(res.json()['message'])


def test_preprocessing_get(task_id : str):

    url = url_service + "Preprocessing/"
    parameters = {'task_id' : task_id, 'token' : 'aaa'}
    res = requests.get(url, params=parameters)

    print(res)
    if res.status_code != 200 :
        print(res.json()['message'])
        return None
    else :
        filename = "preprocessed_trajectories.parquet"
        if "content-disposition" in res.headers.keys():
            filename = re.findall("filename=\"(.+)\"", res.headers['content-disposition'])[0]
        print(f"Writing received file to: {filename}")
        with open(filename, 'wb') as f:
            f.write(res.content)

        #test_file = pd.read_parquet(filename)
        #print(test_file.info())


def test_segmentation_post(pathfile : str) -> Optional[str]:

    url = url_service + "Segmentation/"
    params = {'min_duration_stop': 10, 'max_stop_radius': 0.2, 'token' : 'aaa'}
    files = {'file_trajectories': ('trajectories.parquet', open(pathfile, 'rb'))}

    res = requests.post(url, params=params, files=files)

    print(res)
    print(res.json()['message'])


def test_segmentation_get(task_id : str) :

    url = url_service + "Segmentation/"
    parameters = {'task_id' : task_id, 'token' : 'aaa'}

    res = requests.get(url, params = parameters)
    print(res)
    print(res.status_code)


    if res.status_code != 200 :
        print(f"HTTP status: {res.status_code}")
        if res.status_code == 204: print(f"Task {task_id} is still being processed...")
        else : print(res.json())
        return None
    else :
        # Translate the received stops and moves from json to dataframes.
        stops = pd.DataFrame.from_dict(res.json()['stops'])
        moves = pd.DataFrame.from_dict(res.json()['moves'])

        print(stops.info())
        print(moves.info())

        # Store the dataframes into parquet files.
        stops.to_parquet('stops.parquet')
        moves.to_parquet('moves.parquet')

        def test_segmentation_post(pathfile: str) -> Optional[str]:

            url = url_service + "Segmentation/"
            params = {'min_duration_stop': 10, 'max_stop_radius': 0.2, 'token': 'aaa'}
            files = {'file_trajectories': ('trajectories.parquet', open(pathfile, 'rb'))}

            res = requests.post(url, params=params, files=files)

            print(res)
            if res.status_code != 200:
                print(f"Some error occurred. Code returned by the server: {res.status_code}")
                print(res.json()['message'])
                return None
            else:
                print(f"Message from the server: {res.json()['message']}")
                return res.json()['task_id']


def test_enrichment_post(path_trajs : str,
                         path_moves : str,
                         path_stops : str,
                         path_pois : str,
                         path_social : str,
                         path_weather : str) -> Optional[str]:

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

def test_enrichment_get(task_id: str):

    url = url_service + "Enrichment/"
    parameters = {'task_id': task_id, 'token': 'aaa'}
    res = requests.get(url, params=parameters)

    print(res)
    if res.status_code != 200:
        print(res.json()['message'])
        return None
    else:
        filename = "results.ttl"
        if "content-disposition" in res.headers.keys():
            filename = re.findall("filename=\"(.+)\"", res.headers['content-disposition'])[0]
        print(f"Writing received file to: {filename}")
        with open(filename, 'wb') as f:
            f.write(res.content)


def main() :

    # task_id = test_preprocessing_post('./datasets/rome/rome.parquet')
    # test_preprocessing_get("dcdc83cfd0e245368d2fb4509ccdf5a8")

    # task_id = test_segmentation_post('./preprocessed_trajectories.parquet')
    # test_segmentation_get('723d02b33ef5432daad4aff44ad35e42')

    #test_enrichment_post('./preprocessed_trajectories.parquet', './moves.parquet', './stops.parquet', './datasets/rome/poi/pois.parquet',
    #                     './datasets/rome/tweets/tweets_rome.parquet', './datasets/rome/weather/weather_conditions.parquet')
    test_enrichment_get("637b867f90a04b74a1b53688ec0ffeaf")


if __name__ == '__main__':
    main()