import json

import requests
import pandas as pd
import re


url_service = "http://127.0.0.1:8000/semantic_processor/"


def test_json() :

    url = url_service + "Preprocessing/"
    data = {'name': 'test', 'price': 300}

    res = requests.get(url, json = data)
    print(res.status_code)
    print(res.json())


def test_form() :

    url = url_service + "Preprocessing/"
    files = \
    {
        'num_samples': (None, 1500),
        'speed': (None, 300),
        'compression': (None, True)
    }

    res = requests.get(url, files = files)
    print(res.status_code)
    print(res.json())


def test_preprocessing() :

    url = url_service + "Preprocessing/"
    files = \
    {
        'file_trajectories': ('trajectories.parquet', open('./datasets/rome/rome.parquet', 'rb')),
        'min_num_samples': (None, 1500),
        'max_speed': (None, 300),
        'compress_trajectories': (None, True)
    }
    res = requests.get(url, files = files)


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


def test_segmentation() :

    url = url_service + "Segmentation/"
    files = \
    {
        'file_trajectories': ('trajectories.parquet', open('preprocessed_trajectories.parquet', 'rb')),
        'min_duration_stop': (None, '10'),
        'max_stop_radius': (None, '0.2')
    }

    res = requests.get(url, files = files)
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


def test_enrichment() :

    url = url_service + "Enrichment/"
    files = \
    {
        'file_trajectories': ('trajectories.parquet', open('./preprocessed_trajectories.parquet', 'rb')),
        'file_moves': ('moves.parquet', open('./moves.parquet', 'rb')),
        'file_stops': ('stops.parquet', open('./stops.parquet', 'rb')),
        'file_pois': ('pois.parquet', open('./datasets/rome/poi/pois.parquet', 'rb')),
        'file_social': ('social.parquet', open('./datasets/rome/tweets/tweets_rome.parquet', 'rb')),
        'file_weather': ('weather.parquet', open('./datasets/rome/weather/weather_conditions.parquet', 'rb')),
        'move_enrichment': (None, True),
        'max_dist': (None, 50),
        'dbscan_epsilon': (None, 50),
        'systematic_threshold': (None, 5)
    }
    res = requests.get(url, files = files)


    print(res)
    print(res.status_code)


    filename = "kg.ttl"
    if "content-disposition" in res.headers.keys():
        filename = re.findall("filename=\"(.+)\"", res.headers['content-disposition'])[0]
    print(f"Writing received file to: {filename}")
    with open(filename, 'wb') as f:
        f.write(res.content)


def main() :
    # test_preprocessing()
    # test_segmentation()
    test_enrichment()


if __name__ == '__main__':
    main()