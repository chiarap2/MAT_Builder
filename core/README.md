This folder contains several classes that constitute the core of ```MAT-Builder```'s backend.

The abstract class ```ModuleInterface``` provides the interface that any module
to be used within a semantic enrichment process must implement. For further details
on the interface methods, please see the documentation in ```ModuleInterface```.

The abstract class ```InteractiveModuleInterface``` provides the interface that
any class which aims to expose the functionalities of class that implements ```ModuleInterface```
via a user interface must implement. The abstract class is strictly interconnected with the
class ```InteractivePipeline```. For further details on the interface methods, please see the documentation
in ```InteractiveModuleInterface```.

The class ```InteractivePipeline``` models the notion of semantic enrichment process,
which is concretely seen as a sequence of ```InteractiveModuleInterface```s.
For further information on the methods of such class, please see the documentation
in ```InteractivePipeline```.

The class ```RDFBuilder``` provides the sets of tools necessary to build an RDF knowledge graph according
to the STEPv2 customized ontology present in the ```ontology``` folder.

Finally, the ```modules``` folder contains a mixture of classes that derive the ```ModuleInterface``` and ```InteractiveModuleInterface```
abstract classes, as well as a few other classes.

