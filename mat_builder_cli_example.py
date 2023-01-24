import pandas as pd
import geopandas as gpd

from core.modules import *


### MAIN application ###

def main() :

    print('Executing preprocessing...')
    params_preprocessing = {'trajectories' : pd.read_parquet('./data/Rome/rome.parquet'),
                            'speed' : 300,
                            'n_points' : 1500}
    prepro = Preprocessing()
    prepro.execute(params_preprocessing)


    print('Executing segmentation...')
    params_segmentation = {'trajectories' : prepro.get_results()['preprocessed_trajectories'],
                           'duration' : 10,
                           'radius' : 0.5}
    segm = Segmentation()
    segm.execute(params_segmentation)
    result_segmentation = segm.get_results()


    print('Executing enrichment...')
    enrichment = Enrichment()
    poi_df = gpd.read_parquet('./data/Rome/poi/pois.parquet')
    social_df = pd.read_parquet('./data/tweets/tweets_rome.parquet')
    weather_df = pd.read_parquet('./data/weather/weather_conditions.parquet')
    params_enrichment = {'moves' : result_segmentation['moves'],
                         'move_enrichment' : True,
                         'stops' : result_segmentation['stops'],
                         'poi_place' : 'Rome, Italy',
                         'poi_categories' : None, # ['amenity'],
                         'path_poi' : poi_df,
                         'max_dist' : 50,
                         'social_enrichment' : social_df,
                         "weather_enrichment" : weather_df,
                         'create_rdf' : True}
    enrichment.execute(params_enrichment)


if __name__ == '__main__':
    main()
