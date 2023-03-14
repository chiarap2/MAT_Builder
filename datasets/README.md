This folder contains two archives, each being a multi-file [7Zip](https://7-zip.org/download.html) archive.

- The first archive is ```rome```, which contains a dataset of 26395 trajectories from 3181 individuals (see the root folder of the archive).
The trajectories move over the city of Rome and were collected from 
OpenStreetMap. The archive contains also additional datasets, i.e., the set of
POIs within the province of Rome's boundaries (downloaded from OpenStreetMap) (see the ```poi``` subfolder), 
historical weather information (downloaded from [Meteostat](https://meteostat.net/it/)) (see the ```weather``` subfolder), and a
dataset of social media posts from the individuals which was generated synthetically (see the ```tweets``` subfolder).
All the datasets are ```pandas``` dataframes, exception for the POI dataset which is
a ```geopandas``` DataFrame. All the datasets have been stored according to the parquet format.


- The second archive is ```geolife```, which contains the [GeoLife](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/) dataset.
The dataset contains 17621 trajectories from 178 users. The timestamps of the samples of the trajectories
have been adjusted from the GMT to the GMT+8 timezone. As in the ```rome``` archive case, this archive
contains also a dataset of POIs, a dataset of historical weather information, and a dataset of social media posts
that were generated synthetically.