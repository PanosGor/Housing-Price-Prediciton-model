import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
import os


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

housing = pd.read_csv('housing_mod.txt')
print()
print(housing.describe())
print()
print(pd.DataFrame({'percent_missing': housing.isnull().sum() * 100/ len(housing)}))  

housing.hist(bins=50, figsize=(20,15))

median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)

# ============================== We crate buckets to stratify sample =============================


housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
housing['income_cat'].hist()



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
print()
print(compare_props)

# Dropparw to extra column
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# ======================== California Image ======================

housing = strat_train_set.copy()

california_img=mpimg.imread(r"images\california.png")
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                  s=housing['population']/100, label="Population",
                  c="median_house_value", cmap=plt.get_cmap("jet"),
                  colorbar=False, alpha=0.4)
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar(ticks=tick_values/prices.max())
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)

print()
print('Correlation matrix')
corr_matrix = housing.corr()
#Correlation matrix without added columns
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Corr matrix with added cols
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

df2= pd.get_dummies(housing,columns=['ocean_proximity'],prefix=['ocean_proximity'])

corr_matrix = df2.corr()
print("\n","Correlation matrix with added cols","\n",corr_matrix["median_house_value"].sort_values(ascending=False))



# drop labels for training set # proerxontai apo copy to strat_train_set

# X_train
housing = df2.drop("median_house_value", axis=1) 

# Y_train
housing_labels = df2["median_house_value"].copy()

#==================================================================================

housing_test=strat_test_set.copy()

housing_test["rooms_per_household"] = housing_test["total_rooms"]/housing_test["households"]
housing_test["bedrooms_per_room"] = housing_test["total_bedrooms"]/housing_test["total_rooms"]
housing_test["population_per_household"]=housing_test["population"]/housing_test["households"]

df3= pd.get_dummies(housing_test,columns=['ocean_proximity'],prefix=['ocean_proximity'])

# X_test
housing_test = df3.drop("median_house_value", axis=1) 

# Y_test
housing_test_labels = df3["median_house_value"].copy()

featureNames=list(housing.columns)

# TRAINING MODELS

# ============================== Linear =====================
lin_reg = LinearRegression()
lin_reg.fit(housing, housing_labels)
print("\n","Linear Regression","\n")

lin_mse2 = mean_squared_error(housing_test_labels, lin_reg.predict(housing_test))
lin_rmse2 = np.sqrt(lin_mse2)
print("lin_rmse:",round(lin_rmse2,2))

lin_mae2 = mean_absolute_error(housing_test_labels, lin_reg.predict(housing_test))
print("lin_mae:",round(lin_mae2,2))
print ('R2 score Linear: %.2f'  % r2_score(housing_test_labels, lin_reg.predict(housing_test)))
print("Train set score: ",round(lin_reg.score(housing,housing_labels),2))
print("Test set score: ",round(lin_reg.score(housing_test,housing_test_labels),2))

# ================================== POLYNOMIAL =========================

regr2 = LinearRegression()

polyDegree=2
poly = PolynomialFeatures(degree=polyDegree)


poly_X_train = poly.fit_transform(housing)
poly_X_test = poly.fit_transform(housing_test)

regr2.fit(poly_X_train, housing_labels)

print("\n","Polynomial Regression","\n")
# Explained variance score: 1 is perfect prediction
print ("poly_rmse: %.2f" % np.sqrt(mean_squared_error(housing_test_labels, regr2.predict(poly_X_test))))

print ("poly_mae: %.2f" % mean_absolute_error(housing_test_labels, regr2.predict(poly_X_test)))
print ('R2 score poly: %.2f'  % r2_score(housing_test_labels, regr2.predict(poly_X_test)))

print("Train set score: ",round(regr2.score(poly_X_train,housing_labels),2))
print("Test set score: ",round(regr2.score(poly_X_test,housing_test_labels),2))

# ================================== LASSO =========================


regr3=Lasso(alpha=0.01, max_iter=100000)

regr3.fit(housing, housing_labels)

print("\n","Lasso Regression","\n")
print('Lasso coeficients',regr3.coef_)
print()




print ("Lasso_rmse: %.2f" % np.sqrt(mean_squared_error(housing_test_labels, regr3.predict(housing_test))))

print ("Lasoo_mae: %.2f" % mean_absolute_error(housing_test_labels, regr3.predict(housing_test)))
print ('R2 score Lasso: %.2f'  % r2_score(housing_test_labels, regr3.predict(housing_test)))

print("Train set score: ",round(regr3.score(housing,housing_labels),2))
print("Test set score: ",round(regr3.score(housing_test,housing_test_labels),2))

# ================================== RIDGE 0.1 =========================

regr4=Ridge(alpha=0.1)

regr4.fit(housing, housing_labels)

print("\n","Ridge Regression","\n")
print("Ridge 0.1 coefficients",regr4.coef_)
print()


print ("Ridge_0.1_rmse: %.2f" % np.sqrt(mean_squared_error(housing_test_labels, regr4.predict(housing_test))))

print ("Ridge_0.1_mae: %.2f" % mean_absolute_error(housing_test_labels, regr4.predict(housing_test)))
print ('R2 score Ridge_0.1: %.2f'  % r2_score(housing_test_labels, regr4.predict(housing_test)))

print("Train set score: ",round(regr4.score(housing,housing_labels),2))
print("Test set score: ",round(regr4.score(housing_test,housing_test_labels),2))

# ================================== RIDGE 1 =========================

regr5=Ridge(alpha=1)

regr5.fit(housing, housing_labels)

print()
print("Ridge 1 coefficients",regr5.coef_)
print()
print ('R2 score Ridge_1: %.2f'  % r2_score(housing_test_labels, regr5.predict(housing_test)))
print ("Mean absolute error Ridge_1: %.2f" % mean_absolute_error(housing_test_labels, regr5.predict(housing_test)))
print ("Root Mean squared error Ridge_1: %.2f" % np.sqrt(mean_squared_error(housing_test_labels, regr5.predict(housing_test))))

print("Train set score: ",regr5.score(housing,housing_labels))
print("Test set score: ",regr5.score(housing_test,housing_test_labels))

# ================================== RIDGE 10 =========================


regr6=Ridge(alpha=10)

regr6.fit(housing, housing_labels)

print()
print("Ridge 10 coefficients",regr6.coef_)
print()
print ('R2 score Ridge_10: %.2f'  % r2_score(housing_test_labels, regr6.predict(housing_test)))
print ("Mean absolute error Ridge_10: %.2f" % mean_absolute_error(housing_test_labels, regr6.predict(housing_test)))
print ("Root Mean squared error Ridge_10: %.2f" % np.sqrt(mean_squared_error(housing_test_labels, regr6.predict(housing_test))))

print("Train set score: ",regr6.score(housing,housing_labels))
print("Test set score: ",regr6.score(housing_test,housing_test_labels))


plt.figure(figsize=(8,6))
plt.plot(regr4.coef_, 'o', label="Ridge_0.1")
plt.plot(regr5.coef_, 's', label="Ridge_1")
plt.plot(regr6.coef_, '^', label="Ridge_10")
plt.xticks(range(housing.shape[1]),featureNames,rotation=90)
plt.legend()


plt.figure(figsize=(8,6))
plt.plot(lin_reg.coef_, 'o', label="LinearRegression")
plt.plot(regr3.coef_, 's', label="Lasso")
plt.plot(regr4.coef_, '^', label="Ridge")
plt.xticks(range(housing.shape[1]),featureNames,rotation=90)
plt.legend()


param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing, housing_labels)

print("Grid Search Best Parameters:", grid_search.best_params_)
print("Grid Search Best Estimator:", grid_search.best_estimator_)


final_model=grid_search.best_estimator_

final_predictions = final_model.predict(housing_test)

print("\n","Random Forest Regressor","\n")
final_mse= mean_squared_error(housing_test_labels, final_predictions)
final_rmse=np.sqrt(final_mse)
print("Random Forest Root Mean Squared Error: " ,round(final_rmse,2))
print ("Mean absolute error: %.2f" % mean_absolute_error(housing_test_labels, final_predictions))

print ('R2 score Random Forest: %.2f' % r2_score(housing_test_labels, final_predictions))
print("Train set score: ",round(final_model.score(housing,housing_labels),2))
print("Test set score: ",round(final_model.score(housing_test,housing_test_labels),2))




































plt.show()