# Housing-Price-Prediciton-model
This is a ML regression model made in Python that predicts housing prices in California 


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

*Figure 26: Dataset features *

![image](https://user-images.githubusercontent.com/82097084/166112195-3e785d50-4387-4536-89d1-be2ffa4ed03b.png)

The graphs show that the median income attribute is expressed in US dollars and the numbers represent tens of thousands of dollars. 
Incomes range from 0.5 to 15 in the histogram which represents 5,000 to 150,000 thousand dollars.

Also, the housing median age and median house value have a cap. 
Values above the age of 50 are all set as 50 and median house values above 500,000 are set equal to 500,000.
This could potentially be a serious problem for the algorithm if it is to predict prices above the 500,000 range but in this case it is not an issue. 
This could potentially be a serious problem for the algorithm if it is to predict prices above the 500,000 range but in this case it is not an issue many 
experimenting with the various regression tools are experimented and the dataset does not contain such values.

## Preprocessing 

The first thing that was done was to check for missing values in the dataset. 
The next figure shows that the dataset was very clean with the exception of total_bedrooms feature. 
The way that was decided to solve this problem was to replace the missing values with the median of this column and since the percentage of the missing values is too small it does not mess the data.

*Figure 27:  Missing values percentages*

![image](https://user-images.githubusercontent.com/82097084/166112324-57041fa7-5fe4-4e15-988e-b1d262b46c31.png)

Next the correlation matrix of all features against the dependent (target) variable was made.
I decided to add some new features to the dataset that might give better insights or prove to be better factors for the regression models. 

The new features added where:
-	Rooms per household, that was accomplished by dividing the total rooms with the number of households in the area
-	Bedrooms per room, accomplished by dividing the total bedrooms by the total rooms 
-	Population per household, accomplished dividing population by number of households

Another important step was made here. 
One hot encoding was used in order to break ocean proximity feature that is a categorical one to different columns first so that the regression models 
understand the difference between categorical values and second so that we could see how correlated with the dependent variable the categorical values are.
In the next graphs a correlation matrix of the starting values is shown in the first figure and correlation of the values including the new that were created along 
with the categorical variables is shown

*Figure 28: Correlation of starting values*

![image](https://user-images.githubusercontent.com/82097084/166112382-6c6e804a-8172-4180-a6f1-977368698d7c.png)

*Figure 29: Correlation with new features and one hot encoding*

![image](https://user-images.githubusercontent.com/82097084/166112389-e130bebc-6bee-46fb-a753-33d9b22feb72.png)

Median income is the most correlated value in both cases. 
It is clear that houses that are less than one hour from the ocean are correlated to the median house value and houses that are inland are negatively correlated. 
Also, the new attributes that were added were not all good but rooms per household is better than all the initial features.

The last preprocessing step was creating a train set and a test set. 
The train set was used to train the models and the test set was used for predictions to check how good the models generalize. 
The method that was used for splitting the dataset was stratified sampling. Some testing was done to end up with this decision.  
In order to split the dataset median income was used, as it is the most correlated value, to create buckets that represent the dataset. 
The next figure shows how the bins are. 

After that random splitting and stratified sampling was tested and their errors were compared. 
The results are shows in the next figures.

*Figure 30: Stratified sampling bins*

![image](https://user-images.githubusercontent.com/82097084/166112445-81eb4a08-e115-46eb-8faf-af4b5a99e523.png)

*Figure 31: Stratified vs Random Split*

![image](https://user-images.githubusercontent.com/82097084/166112464-110941b7-c80d-4841-a3fe-99581603f5bb.png)

Overall refers to the proportion of values in the dataset and stratified and random refer to how these proportions end up with each way of splitting. 
It is clear that stratified represents the starting dataset better.

## Linear Regression

Linear regression has no regularization so there is not much you can do to improve this model. 
The only thing possible is to cute features manually until you have the desired result. 
After fitting the model, the results that the team got are depicted in the next figure 

*Figure 32: Linear Regression Results*

![image](https://user-images.githubusercontent.com/82097084/166112523-9462914f-72b3-4f34-9452-43574a6dce3b.png)

In most areas the median_house_values range between $120,000 and $265,000. 
This model has an error of $66913. This means that in some cases it can be up to 50% wrong in its predictions. 
This model is probably underfitting the data.

## Polynomial Regression

The results of the polynomial regression are depicted bellow

*Figure 33: Polynomial Regression Results*

![image](https://user-images.githubusercontent.com/82097084/166112556-3b85b0c7-5ef3-4571-995d-5d76cfdf59c8.png)

Unfortunately the rmse values and test set score shows that this model is completely irrelevant to the dataset 

## Lasso and Ridge 

Lasso and Ridge regression were used also. Lasso and Ridge regression allow for regularization by trying to restrict the weights of the features. 
Unfortunately, since simple linear regression is already underfitting so does Lasso and Ridge. They give almost identical results to simple linear Regression

*Figure 34: Lasso Results*

![image](https://user-images.githubusercontent.com/82097084/166112590-3f74f67c-26d3-4815-b8f7-845e76bd25e5.png)

*Figure 35: Ridge (a=0.1) Results*

![image](https://user-images.githubusercontent.com/82097084/166112598-816c33e9-6872-4913-b07b-8bcdef709cde.png)

*Figure 36: Lasso vs Ridge vs Linear Regression weights*

![image](https://user-images.githubusercontent.com/82097084/166112601-4481160c-3e17-4e01-92ff-e78b82c7e493.png)


The above figure shows the regularization of the weights by Lasso and Ridge compared to Linear Regression 

## Random Forest Regression 

The final model the team decided to run was a random forest regressor. 
Generally, decision trees and forests are very strong models. 
This was perfect for the situation as the previous models all underfitted. 
Grid search was also used to find the optimal model. 

The final results are the following

*Figure 37: Random Forest Results*

![image](https://user-images.githubusercontent.com/82097084/166112662-1a44e41f-8342-4c68-8e94-1e516358e018.png)

This model seems to predict really well, and Root mean Squares Error is significantly small than the rest of the models 

# Instructions 

If the file does not run or break, make sure to adjust in line 23 and line 70 the paths to the correct location in your PC


Run Regression.py and it will give:

- a description of the dataset (count, mean, etc)
- the percentage of missing values in each feature
- a table comparing the percentage of data point in the dataset as far as medial income is conserned and the comparison between
	Random split vs stratified with their errors in percentage
- correlation matrices 
- results of Linear, Polynomial, Lasso, Ridge regressions and a Forest regressor 
