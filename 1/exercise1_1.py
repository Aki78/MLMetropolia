import numpy as np                  # for scientific computing (e.g. culclations with array)
import pandas as pd                 # for data manipulation and analysis
import matplotlib.pyplot as plt     # for visualization


# Download dataset from Github
#!wget https://raw.githubusercontent.com/a-ymst/IntroductionToMachineLearning/main/Datasets/ice_cream_sales.csv

df_icecream = pd.read_csv("ice_cream_sales.csv")  # making a data frame object from csv file
#display(df_icecream)
df_icecream.plot(x="temperature", y="ice cream sales", kind="scatter")




# extract "temperature" column as the input X
X = df_icecream[["temperature"]]

# extract "ice cream sales" column as the target t
t = df_icecream[["ice cream sales"]]

#display(X)
#display(t)





from sklearn.linear_model import LinearRegression

model = LinearRegression()  # model definition (using y=ax+b)
model.fit(X, t)             # optimization (obtaining a and b)

print("a =", model.coef_)
print("b =", model.intercept_)






# convert from dataframe into numpy array
_X = X.to_numpy().reshape(-1)
_t = t.to_numpy().reshape(-1)

print("_X =", _X)
print("_t =", _t)

# V[X]: Variance of X
_X_var = np.sum(_X**2) / _X.shape[0] - np.mean(_X)**2

# COV[X, t]: Covariance of X and t
_Xt_cov = np.sum((_X - np.mean(_X)) * (_t - np.mean(_t))) / (_X.shape[0])

_a = _Xt_cov / _X_var
_b = np.mean(_t) - _a * np.mean(_X)

print("_a =", _a)
print("_b =", _b)






# X0 is a virtual input to draw the regression line
np_X0 = np.arange(-5, 35, 1)                                # [-5, -4, -3, ... ,34]
X0 = pd.DataFrame(data = np_X0, columns=["temperature"])    # convert into pandas DataFrame

# prediction with X0
y0 = model.predict(X0)

# Show a graph
plt.scatter(X, t, color="blue")     #plot dataset
plt.plot(X0, y0, color="black")  #plot regression line
plt.show()



df_gdp = pd.read_csv("gdp_per_capita_finland.csv")
#df_gdp


df_gdp.plot(x="year", y="GDP per capita (USD)", kind="line")




from sklearn.linear_model import LinearRegression

# =====================================================================
# Exercise 1.1.1

X = df_gdp[["year"]]
t = df_gdp[["GDP per capita (USD)"]]

model = LinearRegression()
model.fit(X, t)
y = model.coef_*X + model.intercept_ 

# convert from dataframe into numpy array
_X = X.to_numpy().reshape(-1)
_t = t.to_numpy().reshape(-1)
_y = y.to_numpy().reshape(-1)


print(y)
print(_y)
print("here", _t - _y )

epsilons = _y - _t

ave_epsilon = np.sum(_y - _t)/len(_t)
ave_epsilon2 = np.sum(np.square(_y - _t)/len(_t))


ave_t = np.sum(_t)/len(_t)
ave_t2 = np.sum(np.square(_t)/len(_t))

Vepsilon = ave_epsilon2 - ave_epsilon*ave_epsilon
Vt       = ave_t2 - ave_t*ave_t

R2 = 1 - Vepsilon/Vt
rmse = np.sqrt(np.sqrt(np.dot(_y-_t,_y-_t))/len(_t))
# =====================================================================

# X0 is a virtual input to draw the regression line
np_X0 = np.arange(1950, 2030, 1)                    # [1950, 1951, ... , 2029]
X0 = pd.DataFrame(data = np_X0, columns=["year"])   # convert to pandas DataFrame

y0 = model.predict(X0)

print("a =", model.coef_)
print("b =", model.intercept_)
print("R^2 =", R2)
print("RMSE =", rmse)

# Show a graph
plt.plot(X, t, c="blue")      # plot the dataset
plt.plot(X0, y0, c="black")   # plot the regression line
plt.show()




from sklearn.model_selection import train_test_split

df_gdp = pd.read_csv("gdp_per_capita_finland.csv")
X = df_gdp[["year"]]
t = df_gdp[["GDP per capita (USD)"]]

#Dividing whole dataset into training set and test set
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, random_state=47)

plt.scatter(X_train, t_train, c="blue")
plt.scatter(X_test, t_test, c="orange")
plt.show()


from sklearn.linear_model import LinearRegression

# =====================================================================
# Exercise 1.1.2
# Make linear regression model and fit it with training data.
# Then calculate R^2 score and MSE on both training and test set.

model = LinearRegression()
model.

y_train =
y_test =

R2_train =
R2_test =
rmse_train =
rmse_test =

# =====================================================================

# X0 is a virtual input to draw the regression line
np_X0 = np.arange(1950, 2030, 1)                    # [1950, 1951, ... , 2029]
X0 = pd.DataFrame(data = np_X0, columns=["year"])   # convert into pandas DataFrame
y0 = model.predict(X0)

print("W =", model.coef_)
print("b =", model.intercept_)
print("R^2(train) =", R2_train)
print("R^2(test) =", R2_test)
print("RMSE(train) =", rmse_train)
print("RMSE(test) =", rmse_test)

# show a graph
plt.scatter(X_train, t_train, c="blue")
plt.plot(X0, y0, c="black")
plt.show()

# show a graph
plt.scatter(X_test, t_test, c="orange")
plt.plot(X0, y0, c="black")
plt.show()

