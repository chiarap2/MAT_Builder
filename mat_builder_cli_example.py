import pandas as pd
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
    results = segm.get_results()
    print(results['stops'])
    print(results['moves'])



if __name__ == '__main__':
    main()
