import requests
import pandas as pd
import re


url_service = "http://127.0.0.1:8000/semantic_processor/"


def test_preprocessing(pathfile : str) :

    url = url_service + "Preprocessing/"
    parameters = {'max_speed' : 300, 'min_num_samples' : 1500, 'compress_trajectories' : True}
    files = {'file_trajectories': ('trajectories.parquet', open(pathfile, 'rb'))}
    res = requests.get(url, params = parameters, files = files)


    print(res)
    print(res.status_code)


    filename = "preprocessed_trajectories.parquet"
    if "content-disposition" in res.headers.keys():
        filename = re.findall("filename=\"(.+)\"", res.headers['content-disposition'])[0]
    print(f"Writing received file to: {filename}")
    with open(filename, 'wb') as f:
        f.write(res.content)

    test_file = pd.read_parquet(filename)
    print(test_file.info())


def test_segmentation(pathfile : str) :

    url = url_service + "Segmentation/"
    params = {'min_duration_stop': 10, 'max_stop_radius': 0.2}
    files = {'file_trajectories': ('trajectories.parquet', open(pathfile, 'rb'))}

    res = requests.get(url, params = params, files = files)
    print(res)
    print(res.status_code)

    # Translate the received stops and moves from json to dataframes.
    stops = pd.DataFrame.from_dict(res.json()['stops'])
    moves = pd.DataFrame.from_dict(res.json()['moves'])

    # Translate the datetime fields from string (necessary to represent them in JSON) back to datetime64.
    stops['datetime'] = pd.to_datetime(stops['datetime'])
    stops['leaving_datetime'] = pd.to_datetime(stops['leaving_datetime'])
    moves['datetime'] = pd.to_datetime(moves['datetime'])

    print(stops.info())
    print(moves.info())

    # Store the dataframes into parquet files.
    stops.to_parquet('stops.parquet')
    moves.to_parquet('moves.parquet')


def test_enrichment(path_trajs : str,
                    path_moves : str,
                    path_stops : str,
                    path_pois : str,
                    path_social : str,
                    path_weather : str) :

    url = url_service + "Enrichment/"
    params =\
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
    res = requests.get(url, params = params, files = files)


    print(res)
    print(res.status_code)


    filename = "kg.ttl"
    if "content-disposition" in res.headers.keys():
        filename = re.findall("filename=\"(.+)\"", res.headers['content-disposition'])[0]
    print(f"Writing received file to: {filename}")
    with open(filename, 'wb') as f:
        f.write(res.content)


def main() :
    test_preprocessing('./datasets/rome/rome.parquet')
    test_segmentation('./preprocessed_trajectories.parquet')
    test_enrichment('./preprocessed_trajectories.parquet', './moves.parquet', './stops.parquet', './datasets/rome/poi/pois.parquet', './datasets/rome/tweets/tweets_rome.parquet', './datasets/rome/weather/weather_conditions.parquet')


if __name__ == '__main__':
    main()