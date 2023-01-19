from core.modules import *


### MAIN application ###

def main() :

    print('Executing preprocessing...')
    params_preprocessing = {'path' : './data/Rome/rome.parquet',
                            'speed' : 300,
                            'n_points' : 3000}
    prepro = Preprocessing()
    prepro.execute(params_preprocessing)
    results = prepro.get_results()
    print(results['preprocessed_trajectories'])


    print('Executing segmentation...')
    params_segmentation = {'trajectories' : results['preprocessed_trajectories'],
                           'duration' : 10,
                           'radius' : 0.5}
    segm = Segmentation()
    segm.execute(params_segmentation)
    results = segm.get_results()
    print(results['stops'])
    print(results['moves'])



if __name__ == '__main__':
    main()
