import json

import requests
import pandas as pd


def test_json() :

    url = "http://127.0.0.1:8000/Preprocessing/"
    data = {'name': 'test', 'price': 300}

    res = requests.get(url, json = data)
    print(res.status_code)
    print(res.json())


def test_form() :

    url = "http://127.0.0.1:8000/Preprocessing/"
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

    url = "http://127.0.0.1:8000/semantic/Preprocessing/"
    files = \
    {
        'file_trajectories': ('trajectories.parquet', open('test_3.parquet', 'rb')),
        'min_num_samples': (None, 1500),
        'max_speed': (None, 300),
        'compress_trajectories': (None, True)
    }

    res = requests.get(url, files = files)
    print(res.status_code)
    with open('preprocessed_trajectories.parquet', 'wb') as f:
        f.write(res.content)

    prova = pd.read_parquet('preprocessed_trajectories.parquet')
    print(prova.info())


def test_segmentation() :

    url = "http://127.0.0.1:8000/semantic/Segmentation/"
    files = \
    {
        'file_trajectories': ('trajectories.parquet', open('preprocessed_trajectories.parquet', 'rb')),
        'min_duration_stop': (None, '10'),
        'max_stop_radius': (None, '0.2')
    }

    res = requests.get(url, files = files)
    print(res)
    print(res.status_code)

    stops = pd.DataFrame.from_dict(res.json()['stops'])
    moves = pd.DataFrame.from_dict(res.json()['moves'])

    print(stops)
    print(stops.info())
    print(moves)
    print(moves.info())

    stops.to_parquet('stops.parquet')
    moves.to_parquet('stops.parquet')


def main() :
    # test_json()
    # test_form()
    test_preprocessing()
    test_segmentation()


if __name__ == '__main__':
    main()