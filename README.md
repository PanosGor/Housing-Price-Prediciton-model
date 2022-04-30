# Housing-Price-Prediciton-model
This is a ML regression model that predicts housing prices in California 


## Introduction

The dataset that was used for the experiments is about housing prices in California. 
It contains 20640 data points and 9 features (longitude, latitude, housing median age, total rooms, total bedrooms, population of the area, households in that area, 
median income of citizens in that area, ocean proximity of the house and the dependent variable which is median house value. 
All the attributes are numerical except Ocean proximity which is categorical.

||longitude|latitude|Housing median age|Total rooms|Total bedrooms|population|households|Median income|Median house value|Ocean proximity|
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
|0|-122.23|37.88|41.0|880.0|129.0|322.0|126.0|8.3252|452600.0|NEAR BAY|
|1|-122.22|37.86|21.0|7099.0|1106.0|2401.0|1138.0|8.3014|358500.0|NEAR BAY|
|2|-122.24|37.85|52.0|1467.0|190.0|496.0|177.0|7.2574|352100.0|NEAR BAY|
|3|-122.25|37.85|52.0|1274.0|235.0|558.0|219.0|5.6431|341300.0|NEAR BAY|
|4|-122.25|37.85|52.0|1627.0|280.0|565.0|259.0|3.8462|342200.0|NEAR BAY|

Since the dataset has longitude and latitude the next visual was created to see how the areas are distributed in California. 

*Figure 25: California with the various housing areas*

![image](https://user-images.githubusercontent.com/82097084/166112164-054f3b46-7d7c-4e8b-9739-b582232f5b28.png)

Next, histograms about all features was created so that it is easy to see how the data are distributed and if some useful insights can be derived.

![image](https://user-images.githubusercontent.com/82097084/166112195-3e785d50-4387-4536-89d1-be2ffa4ed03b.png)

