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

To use ```MAT-Builder``` with the graphical interface, run ```mat_builder.py```


## **Installation procedure**

MAT-Builder consists of a set of Python scripts (plus a set of additional assets) which make exclusively use of open-source libraries. In the following we illustrate the installation procedure needed to execute MAT-Builder. The installation procedure has been tested on Windows 10, Ubuntu (version > 20.x), and macOS.

1. The first step requires installing a Python distribution that includes a package manager. To this end we recommend installing [Anaconda](https://www.anaconda.com/products/distribution), a cross-platform Python package manager and environment-management system which satisfies the above criteria.

2. Once Anaconda has been installed, the next step requires to set up a virtual environment containing the open-source libraries that MAT-Builder requires during its execution. To this end we provide a YAML file, ```mat_builder.yml```, that can be used to set the environment up. More precisely, the user must first open an Anaconda powershell prompt. Then, the user must type in the prompt ```conda env create -f path\mat_builder.yml -n name_environment```, where ```path``` represents the path in which ```mat_builder.yml``` is located, while ```name_environment``` represents the name the user wants to assign to the virtual environment.

3.	Once the environment has been created, the user must activate it in the prompt by typing ```conda activate name_environment```. The user will be now able to execute and use MAT-Builder.



### **Usage**
---

To use ```MAT-Builder``` the raw trajectory dataset must have the following data:
- trajectory ID
- user ID
- latitude
- longitude
- timestamp

``MAT-Builder`` is organized in modules. Each module corresponds to a step of the ***semantic enrichment process*** (i.e., preprocessing, segmentation, and enrichment). Users can of course customize the entire process. 

The ``preprocessing`` module allows users to:
- remove outliers
- remove trajectories with few points
- compress trajectories

The ``segmentation`` module allows users to find ***stops*** and ***moves***.

The ```Enrichment``` module allows practioners to enrich different "entities" of a trajectory:
- **Stop enrichment**: 
    - categorize stops into:
        - *systematic stops*: stops that fall in the same area more than a given number of times. They are enriched as *Home*, *Work* or *Other*.
        - *occasional stops*: stops that are not systematic. They are enriched with the most nearest PoIs. PoI dataset must have an ID, latitude, longitude, and a category at least. If users don't have a PoI dataset, they can download them from OpenStreetMap.
- **Move enrichment**: users can enrich moves with the transportation mean
- **Trajectory enrichment**: users can enrich the entire trajectory with the weather conditions that must have an ID, the Trajectory ID and the timestamp. 
- **User enrichment** : users can enrich the 'users entity' with the social media posts. These data must have an ID, the user ID and the timestamp. If social media posts are geolocated, the enrichment could be based on spatial information (such as for occasional stops).

### **Citing us**
---
If you use MAT-Builder, please cite the following paper:

C. Pugliese, F. Lettich, C. Renso, F. Pinelli, Mat-builder: a system to build semantically
enriched trajectories, in: The 23rd IEEE International Conference on Mobile Data Manage-
ment, Cyprus, 2022

```
@inproceedings{Pugliese22,
author = {Chiara Pugliese  and Francesco Lettich  and Chiara Renso and Fabio Pinelli},
title = { MAT-Builder: a System to Build Semantically Enriched Trajectories},
booktitle = {The 23rd IEEE International Conference on Mobile Data Management, Cyprus},
year = {2022}
}
```
