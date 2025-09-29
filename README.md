# Wheat yield forecasts with seasonal climate models and long short-term memory networks

Code to reproduce [our study from Computers and Electronics in Agriculture](https://doi.org/10.1016/j.compag.2025.110965)

### Crop calendar and study region

For the USA, the Great Plains, inlcuding South Dakota, Nebraska, Colorado, Kansas, Oklahoma and Texas is the heartland of winter wheat production (10). Great plains states are highlighted with a blue border in the figure below. 

![](data_preparation/figures/season_wheat_US.png)

### Annual state-level yield and trend

Yield trend in the great plains from 2002-2022 is neglectable. Annual yield data is not detrended, we train a model to estimate absolute yield instead of yield anomalies.

![](data_preparation/figures/yield_trend_US_wheat.png)


### Citation
```
@article{ZACHOW2025110965,
title = {Wheat yield forecasts with seasonal climate models and long short-term memory networks},
journal = {Computers and Electronics in Agriculture},
volume = {239},
pages = {110965},
year = {2025},
issn = {0168-1699},
doi = {https://doi.org/10.1016/j.compag.2025.110965},
url = {https://www.sciencedirect.com/science/article/pii/S0168169925010713},
author = {Maximilian Zachow and Stella Ofori-Ampofo and Harald Kunstmann and Rıdvan Salih Kuzu and Xiao Xiang Zhu and Senthold Asseng},
keywords = {Seasonal climate models, Crop yield, Wheat, Agriculture, LSTM}
}



