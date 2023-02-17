from dash import Dash

from core.modules import *
from core.InteractivePipeline import InteractivePipeline


### MAIN application ###

def main() :    
    
    # Instantiate the Dash application.
    app = Dash(__name__)


    # By default, Dash applies validation to your callbacks, which performs checks such as validating the types of callback arguments and checking to see whether the specified Input and Output components actually have the specified properties. For full validation, all components within your callback must exist in the layout when your app starts, and you will see an error if they do not.
    # However, in the case of more complex Dash apps that involve dynamic modification of the layout (such as multi-page apps), not every component appearing in your callbacks will be included in the initial layout. You can remove this restriction by disabling callback validation like this:
    app.config.suppress_callback_exceptions = True


    # Object representing the pipeline to be executed.
    modules_pipeline = [InteractivePreprocessing,
                        InteractiveSegmentation,
                        InteractiveEnrichment]
                        
    pipeline = InteractivePipeline(app, modules_pipeline)


    app.run_server(debug=True)



if __name__ == '__main__':
    main()