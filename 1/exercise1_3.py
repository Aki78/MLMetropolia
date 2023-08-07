import numpy as np                  # for scientific computing (e.g. culclations with array)
import pandas as pd                 # for data manipulation and analysis
import matplotlib.pyplot as plt     # for visualization
import seaborn as sns               # for visualization


df_icecream = pd.read_csv("ice_cream_sales.csv")    # making data frame object from csv file


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# Standardization
scaler = StandardScaler()
scaler.fit(df_icecream[["temperature"]])
std_X = scaler.transform(df_icecream[["temperature"]])  # standardization for the input

# Extention to polynomial
poly_degree = 5
pf = PolynomialFeatures(degree=poly_degree, include_bias=False)
pf.fit(std_X)
X = pf.transform(std_X)                             # Note: fit_transform() provides numpy array
t = df_icecream[["ice cream sales"]].to_numpy()     # Note: convert into numpy array




from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge      # Ridge regression model
from sklearn.linear_model import Lasso      # Lasso regression model

# Model definition (Choose one of the models below)
#model = LinearRegression(fit_intercept = True)      # model definition (using y = a0*x + a1*x^2 + b)
#model = Ridge(alpha = 0.5, fit_intercept = True)    # model definition
model = Lasso(alpha = 100.0, fit_intercept = True)    # model definition

# Optimizaing the parameters
# Note: X and t are numpy array. Not pandas DataFrame objects
model.fit(X, t)                                     # fit() can accept both numpy array and pandas DataFrame
y = model.predict(X)
R2 = model.score(X, t)
rmse = np.sqrt(np.average((t-y)**2))

print("a =", model.coef_)
print("b =", model.intercept_)
print("R^2 =", R2)
print("RMSE =", rmse)

# Drawing the regression line
np_X0 = np.arange(-5, 35, 1).reshape(-1,1)                  # [-5, -4, -3, ... ,34]
df_X0 = pd.DataFrame(data = np_X0, columns=["temperature"]) # convert into pandas DataFrame
std_X0 = scaler.transform(df_X0)        # standardization
X0 = pf.transform(std_X0)               # expand temperature to polynomial
y0 = model.predict(X0)

plt.scatter(X[:,0], t, c="blue")        # X[:,0] means the first column of X
plt.plot(X0[:,0], y0, c="black")     # X0[:,0] means the first column of X0
plt.show()


df_gdp = pd.read_csv("gdp_per_capita_finland.csv")
df_gdp.plot(x="year", y="GDP per capita (USD)", kind="line")














from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dataset separation
X = df_gdp[["year"]]
t = df_gdp[["GDP per capita (USD)"]]
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, random_state=47)

# Standardization
scaler = StandardScaler()
scaler.fit(X_train)
std_X_train = scaler.transform(X_train)
std_X_test = scaler.transform(X_test)

print("mean = ", scaler.mean_)
print("std = ", np.sqrt(scaler.var_))

plt.scatter(std_X_train, t_train, c="blue")
plt.scatter(std_X_test, t_test, c="orange")
plt.show()

# Extention to polynomials
poly_degree = 4
pf = PolynomialFeatures(degree=poly_degree, include_bias=False)
pf.fit(std_X_train)
X_train = pf.transform(std_X_train)
X_test = pf.transform(std_X_test)

print("X_train.shape =", X_train.shape)
print("X_test.shape =", X_test.shape)







from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# Model definition (Choose one of the models below)
model = LinearRegression(fit_intercept = True)      # model definition (using y = a0*x + a1*x^2 + b)
#model = Ridge(alpha = 10, fit_intercept = True)    # model definition
#model = Lasso(alpha = 10, fit_intercept = True)    # model definition

# Parameter optimization and prediction
model.fit(X_train, t_train)
y_train = model.predict(X_train)
y_test = model.predict(X_test)

# Evaluation of the model
R2_train =  model.score(X_train, t_train)
R2_test =  model.score(X_test, t_test)
rmse_train = np.sqrt(np.average((t_train - y_train)**2))
rmse_test = np.sqrt(np.average((t_test - y_test)**2))

print("W =", model.coef_)
print("b =", model.intercept_)
print("R^2_train =", R2_train)
print("R^2_test =", R2_test)
print("RMSE(train) =", rmse_train)
print("RMSE(test) =", rmse_test)


# Drawing the regression line
np_X0 = np.arange(1950, 2030, 1)                        # [1950, 1951, ... , 2029]
df_X0 = pd.DataFrame(data = np_X0, columns=["year"])    # Convert into pandas DatFrame
std_X0 = scaler.transform(df_X0[["year"]])              # Standardization for X0
X0 = pf.transform(std_X0)                               # extend X0 to polynomial
y0 = model.predict(X0)

plt.scatter(X_train[:,0], t_train, c="blue")    # plotting the training dataset
plt.plot(X0[:,0], y0, c="black")                # drawing the regression line
plt.show()

plt.scatter(X_test[:,0], t_test, c="orange")    # plotting the training dataset
plt.plot(X0[:,0], y0, c="black")                # drawing the regression line
plt.show()





#================== Exercise 1.3 Ridge ==========================
from sklearn.linear_model import Ridge

# Model definition (Choose one of the models below)
#model = LinearRegression(fit_intercept = True)      # model definition (using y = a0*x + a1*x^2 + b)
model = Ridge(alpha = 10, fit_intercept = True)    # model definition
#model = Lasso(alpha = 10, fit_intercept = True)    # model definition

# Parameter optimization and prediction
model.fit(X_train, t_train)
y_train = model.predict(X_train)
y_test = model.predict(X_test)

# Evaluation of the model
R2_train =  model.score(X_train, t_train)
R2_test =  model.score(X_test, t_test)
rmse_train = np.sqrt(np.average((t_train - y_train)**2))
rmse_test = np.sqrt(np.average((t_test - y_test)**2))

print("Ridge W =", model.coef_)
print("Ridge b =", model.intercept_)
print("Ridge R^2_train =", R2_train)
print("Ridge R^2_test =", R2_test)
print("Ridge RMSE(train) =", rmse_train)
print("Ridge RMSE(test) =", rmse_test)


# Drawing the regression line
np_X0 = np.arange(1950, 2030, 1)                        # [1950, 1951, ... , 2029]
df_X0 = pd.DataFrame(data = np_X0, columns=["year"])    # Convert into pandas DatFrame
std_X0 = scaler.transform(df_X0[["year"]])              # Standardization for X0
X0 = pf.transform(std_X0)                               # extend X0 to polynomial
y0 = model.predict(X0)

plt.scatter(X_train[:,0], t_train, c="blue")    # plotting the training dataset
plt.plot(X0[:,0], y0, c="black")                # drawing the regression line
plt.show()

plt.scatter(X_test[:,0], t_test, c="orange")    # plotting the training dataset
plt.plot(X0[:,0], y0, c="black")                # drawing the regression line
plt.show()



#================== Exercise 1.3 Lasso ==========================
from sklearn.linear_model import Ridge

# Model definition (Choose one of the models below)
#model = LinearRegression(fit_intercept = True)      # model definition (using y = a0*x + a1*x^2 + b)
#model = Ridge(alpha = 10, fit_intercept = True)    # model definition
model = Lasso(alpha = 10, fit_intercept = True)    # model definition

# Parameter optimization and prediction
model.fit(X_train, t_train)
y_train = model.predict(X_train)
y_test = model.predict(X_test)

# Evaluation of the model
R2_train =  model.score(X_train, t_train)
R2_test =  model.score(X_test, t_test)
#rmse_train = np.sqrt(np.average(np.abs(t_train - y_train))) ???
#rmse_test = np.sqrt(np.average(np.abs(t_test - y_test))) ???
##
#print("Lasso W =", model.coef_)
#print("Lasso b =", model.intercept_)
#print("Lasso R^2_train =", R2_train)
#print("Lasso R^2_test =", R2_test)
#print("Lasso RMSE(train) =", rmse_train)
#print("Lasso RMSE(test) =", rmse_test)
#
#
## Drawing the regression line
#np_X0 = np.arange(1950, 2030, 1)                        # [1950, 1951, ... , 2029]
#df_X0 = pd.DataFrame(data = np_X0, columns=["year"])    # Convert into pandas DatFrame
#std_X0 = scaler.transform(df_X0[["year"]])              # Standardization for X0
#X0 = pf.transform(std_X0)                               # extend X0 to polynomial
#y0 = model.predict(X0)
#
#plt.scatter(X_train[:,0], t_train, c="blue")    # plotting the training dataset
#plt.plot(X0[:,0], y0, c="black")                # drawing the regression line
#plt.show()
#
#plt.scatter(X_test[:,0], t_test, c="orange")    # plotting the training dataset
#plt.plot(X0[:,0], y0, c="black")                # drawing the regression line
#plt.show()



