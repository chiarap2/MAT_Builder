import pandas as pd
import geopandas as gpd

from core import Preprocessing, Segmentation, Enrichment


### MAIN application ###

def main() :

    print('Executing preprocessing...')
    params_preprocessing = {'trajectories' : pd.read_parquet('./datasets/rome/rome.parquet'),
                            'speed' : 300,
                            'n_points' : 1500,
                            'compress' : True}
    prepro = Preprocessing()
    prepro.execute(params_preprocessing)


    print('Executing segmentation...')
    params_segmentation = {'trajectories' : prepro.get_results()['preprocessed_trajectories'],
                           'duration' : 10,
                           'radius' : 0.2}
    segm = Segmentation()
    segm.execute(params_segmentation)
    result_segmentation = segm.get_results()


    print('Executing enrichment...')
    enrichment = Enrichment()
    poi_df = gpd.read_parquet('./datasets/rome/poi/pois.parquet')
    social_df = pd.read_parquet('./datasets/rome/tweets/tweets_rome.parquet')
    weather_df = pd.read_parquet('./datasets/rome/weather/weather_conditions.parquet')
    params_enrichment = {'trajectories' : result_segmentation['trajectories'],
                         'moves' : result_segmentation['moves'],
                         'move_enrichment' : True,
                         'stops' : result_segmentation['stops'],
                         'poi_place' : 'Rome, Italy',             # IGNORED, if path_poi is not None.
                         'poi_categories' : None, # ['amenity'],  # IGNORED, if path_poi is not None.
                         'path_poi' : poi_df,
                         'max_dist' : 50,
                         'dbscan_epsilon' : 50,
                         'systematic_threshold' : 5,
                         'social_enrichment' : social_df,
                         "weather_enrichment" : weather_df,
                         'create_rdf' : True}
    enrichment.execute(params_enrichment)
    enrichment.get_results()['rdf_graph'].serialize_graph('kg.ttl')


if __name__ == '__main__':
    main()
