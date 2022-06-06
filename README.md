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

### **Usage**
---

To use ```MAT-Builder``` your trajectory dataset must have the following data:
- trajectory ID
- user ID
- latitude
- longitude
- timestamp

``MAT-Builder`` is organized in module. Each module corresponds to a step of the ***semantic enrichment process*** (i.e., preprocessing, segmentation, and enrichment). Users can customize the entire process. 

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


### **Requirement**
---
To run MAT-Builder install requirements.

```pip install requirements.txt```

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