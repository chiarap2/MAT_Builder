## **MAT-Builder: a System to Build Semantically Enriched Trajectories**

---
The notion of **multiple aspect trajectory** (MAT) has
been recently introduced in the literature to represent movement
data that is heavily semantically enriched with dimensions
(aspects) representing various types of semantic information (e.g.,
stops, moves, weather, traffic, events, and points of interest).
Aspects may be large in number, heterogeneous, or structurally
complex. Although there is a growing volume of literature
addressing the modelling and analysis of multiple aspect tra-
jectories, the community suffers from a general lack of publicly
available datasets. This is due to privacy concerns that make it
difficult to publish such type of data, and to the lack of tools that
are capable of linking raw spatio-temporal data to different types
of semantic contextual data. In this work we aim to address this
last issue by presenting ```MAT-Builder```, a system that not only
supports users during the whole semantic enrichment process,
but also allows the use of a variety of external data sources.
Furthermore, ```MAT-Builder``` has been designed with modularity
and extensibility in mind, thus enabling practitioners to easily add
new functionalities to the system and set up their own semantic
enrichment processes.

## **Installation procedure**

MAT-Builder consists of a set of Python scripts (plus a set of additional assets) which make exclusively use of open-source libraries. In the following we illustrate the installation procedure needed to execute MAT-Builder. The installation procedure has been tested on Windows 10, Ubuntu (version > 20.x), and macOS.

1. The first step requires installing a Python distribution that includes a package manager. To this end we recommend installing [Anaconda](https://www.anaconda.com/products/distribution), a cross-platform Python package manager and environment-management system which satisfies the above criteria.

2. Once Anaconda has been installed, the next step requires to set up a virtual environment containing the open-source libraries that MAT-Builder requires during its execution. To this end we provide a YAML file, ```mat_builder.yml```, that can be used to set the environment up. More precisely, the user must first open an Anaconda powershell prompt. Then, the user must type in the prompt ```conda env create -f path\mat_builder.yml -n name_environment```, where ```path``` represents the path in which ```mat_builder.yml``` is located, while ```name_environment``` represents the name the user wants to assign to the virtual environment.

3.	Once the environment has been created, the user must activate it in the prompt by typing ```conda activate name_environment```. The user will be now able to execute and use MAT-Builder.



## **Usage**

To use ```MAT-Builder``` with the graphical interface, run ```mat_builder_ui_example.py```



### **MAT-building pipeline** and **modules**
---

``MAT-Builder`` revolves around the notion of ***MAT-building pipeline***, which is a
semantic enrichment process orchestrated conducted according to a sequence of steps. Each step
represents a specific macro-task and is implemented via a module that extends the
``InteractiveModuleInterface`` abstract class. Currently, there are three modules that have been
included in ``MAT-Builder``'s current version: ***trajectory preprocessing***, ***trajectory 
segmentation***, and ***enrichment***. To see how the modules can be used to set up a MAT-building pipeline please see
the script ``mat_builder_ui_example.py``. 


The ``InteractivePreprocessing`` module implements the trajectory preprocessing step. It takes in input a dataset of raw trajectories and let users:
- remove outliers
- remove trajectories with few points
- compress trajectories

The ``InteractivePreprocessing`` requires the raw trajectory dataset to be stored in a pandas DataFrame,
stored in the Parquet format, and have the following columns:
- ```traj_id```: trajectory ID (string)
- ```user```: user ID (integer)
- ```lat```: latitude of a trajectory sample (float)
- ```lon```: longitude of a trajectory sample (float)
- ```time```: timestamp of a sample (datetime64)

The ``InteractiveSegmentation`` module implements the trajectory segmentation step. It takes in input a set of preprocessed trajectories, and segments
each trajectory into ***stop*** and ***move segments***.

The ```InteractiveEnrichment``` module takes in input the preprocessed trajectories, as well as their stop and move segments, and enriches trajectories and trajectory users with aspects
(or semantic dimensions). The aspects currently supported by the module are as follows:
- **Regularity**: stop segments are categorized into:
  - *systematic stops*: stops that fall in the same area more than a given number of times. They are augmented with the labels  *Home*, *Work* or *Other*.
  - *occasional stops*: stops that are not systematic.
    
  Both occasional and systematic stops are augmented with the nearest POIs. 
  The POI dataset used to augment the stops can either be downloaded from OpenStreetMap (not recommended, this operation might be quite slow),
  or supplied via a local file. In the latter case, the POI dataset must be stored in a GeoDataFrame, according to the Parquet format, and must
  have the following columns:
  - ```osmid```: POI OSM identifier (integer)
  - ```element_type```: POI OSM element type (string)
  - ```name```: POI native name (string)
  - ```name:en```: POI English name (string) 
  - ```wikidata```: POI WikiData identifier (string)
  - ```geometry```: POI geometry (GeoPandas geometry object)
  - ```category```: POI category (string)
  For viable examples of POI datasets, please have a look at the datasets in the ```datasets``` folder.


- **Move**: trajectories are enriched with the move segments. The segments can also be augmented with the transportation mean probably used.


- **Weather**: trajectories are enriched with weather conditions. Such information must be provided via a pandas DataFrame in the form of daily weather conditions, stored according to the 
  Parquet format, and must have the following columns:
  - ```DATE```: date in which the weather observation was recorded (string or datetime64).
  - ```TAVG_C```: average temperature in celsius (float).
  - ```DESCRIPTION```: weather conditions (string).
  For viable examples of weather conditions datasets, please look at the datasets in the ```datasets``` folder.


- **Social media** : trajectory users are enriched with their social media posts. Social media data must be provided via a pandas DataFrame stored according to 
  the Parquet format and must have the following columns:
  - ```tweet_ID```: ID of the tweet (integer)
  - ```text```: post text (string)
  - ```tweet_created```: timestamp of the tweet (datetime64)
  - ```uid```: identifier of the user who posted the tweet.



## **Datasets**

For more details on the datasets, please have a look at the ```datasets``` folder.

## **Citing us**

If you use MAT-Builder, please cite the following papers:

C. Pugliese, F. Lettich, C. Renso, F. Pinelli, Mat-builder: a system to build semantically
enriched trajectories, in: The 23rd IEEE International Conference on Mobile Data Management, Cyprus, 2022

```
@inproceedings{Pugliese22,
author = {Chiara Pugliese  and Francesco Lettich  and Chiara Renso and Fabio Pinelli},
title = { MAT-Builder: a System to Build Semantically Enriched Trajectories},
booktitle = {The 23rd IEEE International Conference on Mobile Data Management, Cyprus},
year = {2022}
}
```