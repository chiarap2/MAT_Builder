from core.modules import *


### MAIN application ###

def main() :

    # Object representing the pipeline to be executed.
    # TODO: To be activated once the Pipeline class is ready.

    # modules_pipeline = [InteractivePreprocessing,
    #                    InteractiveSegmentation,
    #                    InteractiveEnrichment]
                        
    # pipeline = InteractivePipeline(app, modules_pipeline)


    params_preprocessing = {'path' : './data/Rome/rome.parquet',
                            'speed' : 300,
                            'n_points' : 3000}
    prepro = Preprocessing()
    prepro.execute(params_preprocessing)
    results = prepro.get_results()['preprocessed_trajectories']



if __name__ == '__main__':
    main()
