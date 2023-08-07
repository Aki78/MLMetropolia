import numpy as np                  # for scientific computing (e.g. culclations with array)
import pandas as pd                 # for data manipulation and analysis
import matplotlib.pyplot as plt     # for visualization


df_icecream = pd.read_csv("ice_cream_sales.csv")    # making data frame object from csv file


X = df_icecream[["temperature"]]            # extracting "temperature" column as input X
X["temperature^2"] = X["temperature"]**2    # adding "temperature^2" column to the input X

t = df_icecream[["ice cream sales"]]        # extracting 'ice cream sales' column as target t

#display(X)
#display(t)

from sklearn.linear_model import LinearRegression

# Definition of a linear regression model, optimization and prediction
model = LinearRegression(fit_intercept = True)  # If fit_intercept = True, the intercept b will be used
model.fit(X, t)         # optimization (obtaining a0, a1 and b)
y = model.predict(X)    # prediction

# Model evaluation
R2 = model.score(X,t)
rmse = np.sqrt(np.average((t-y)**2))

print("A =", model.coef_)       # This case, there are 2 parameters: a[0] and a[1]
print("b =", model.intercept_)
print("R^2 =", R2)
print("RMSE=", rmse)
     

# X0 is a virtual input to draw the regression line
np_X0 = np.arange(-5, 35, 1)                                # [-5, -4, -3, ... ,34]
X0 = pd.DataFrame(data = np_X0, columns=["temperature"])    # convert into pandas DataFrame
X0["temperature^2"] = X0["temperature"]**2                  # adding the squared temperature to the dataframe

#display(X0)




y0 = model.predict(X0)        # prediction for the virtual input

# Drawing the regression graph
plt.scatter(X["temperature"], t, c="blue")      #plot the originala dataset (There are 6 samples)
plt.plot(X0[["temperature"]], y0, c="black")    #plot the regression line
plt.show()



#=============== Exercise 1.2.1 ====================


from sklearn.linear_model import LinearRegression



X = df_icecream[["temperature"]]            # extracting "temperature" column as input X
X["temperature^2"] = X["temperature"]**2    # adding "temperature^2" column to the input X
X["temperature^3"] = X["temperature"]**3    # adding "temperature^2" column to the input X

# Definition of a linear regression model, optimization and prediction
model = LinearRegression(fit_intercept = True)  # If fit_intercept = True, the intercept b will be used
model.fit(X, t)         # optimization (obtaining a0, a1 and b)
y = model.predict(X)    # prediction

# Model evaluation
R2 = model.score(X,t)
rmse = np.sqrt(np.average((t-y)**2))

print("A =", model.coef_)       # This case, there are 2 parameters: a[0] and a[1]
print("b =", model.intercept_)
print("R^2 =", R2)
print("RMSE=", rmse)


#====================================================


from sklearn.preprocessing import PolynomialFeatures

# Defining polynomial features and fitting it to the test set
poly_degree = 3
pf = PolynomialFeatures(degree=poly_degree, include_bias=False)  # If inlude_bias=True, output includes constant term
pf.fit(df_icecream[["temperature"]])

print(pf.degree)
print(pf.feature_names_in_)





# Extention to polynomials
X = pf.transform(df_icecream[["temperature"]])    # pf.transform() provides numpy array
t = df_icecream[["ice cream sales"]].to_numpy()   # convert into numpy array

#display(X)
#display(t)
     




# Optimizaing the parameters
# Note: X and t are numpy array. Not pandas DataFrame objects
model = LinearRegression(fit_intercept = True)      # model definition (using y = a0*x + a1*x^2 + b)
model.fit(X, t)                                     # fit() can accept both numpy array and pandas DataFrame

# Prediction
y = model.predict(X)

# Model evaluation
R2 = model.score(X,t)
rmse = np.sqrt(np.average((t-y)**2))

print("a =", model.coef_)
print("b =", model.intercept_)
print("R^2 =", R2)
print("RMSE =", rmse)


# Drawing the regression line
np_X0 = np.arange(-5, 35, 1)                            # X0 = [-5, -4, -3, ... ,34]
X0 = pd.DataFrame(data=np_X0, columns=["temperature"])  # Convert into pandas DataFrame

X0 = pf.transform(X0)   # Extension X0 to polynomial
y0 = model.predict(X0)  # Prediction

plt.scatter(X[:,0], t, c="blue")    # X[:,0] means the first column of X
plt.plot(X0[:,0], y0, c="black")    # X0[:,0] means the first column of X0
plt.show()






df_gdp = pd.read_csv("gdp_per_capita_finland.csv")
#display(df_gdp)
df_gdp.plot(x="year", y="GDP per capita (USD)", kind="line")







from sklearn.model_selection import train_test_split

X = df_gdp[["year"]]
t = df_gdp[["GDP per capita (USD)"]]

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 0.3, random_state = 47)

print("X_train.shape = ", X_train.shape)
print("X_test.shape = ", X_test.shape)
plt.scatter(X_train, t_train, c="blue")
plt.scatter(X_test, t_test, c="orange")








from sklearn.preprocessing import StandardScaler

# Defining a a standardization scaler and fitting it to the training set
scaler = StandardScaler()
scaler.fit(X_train)      # Scaler should be fit to the training set

print("mean = ", scaler.mean_)
print("std = ", np.sqrt(scaler.var_))


# Standardization for the training and the test set
std_X_train = scaler.transform(X_train)
std_X_test = scaler.transform(X_test)

plt.scatter(std_X_train, t_train, c="blue")
plt.scatter(std_X_test, t_test, c="orange")









# For saving the results
df_results = pd.DataFrame(columns=["R^2", "RMSE(train)", "RMSE(test)"])


from sklearn.linear_model import LinearRegression

# Definition of PolynomialFeatures and fitting it
poly_degree = 1
pf = PolynomialFeatures(degree=poly_degree, include_bias=False)
pf.fit(std_X_train)

# Extention to polynomials
X_train = pf.transform(std_X_train)
X_test = pf.transform(std_X_test)

# Model definition and optimization
model = LinearRegression(fit_intercept = True)
model.fit(X_train, t_train)   # Parameter optimization

# Prediction
y_train = model.predict(X_train)
y_test = model.predict(X_test)

# Model evaluation
R2 = model.score(X_train, t_train)
rmse_train = np.sqrt(np.average((t_train - y_train)**2))    # calclate RMSE on train dataset
rmse_test = np.sqrt(np.average((t_test - y_test)**2))       # calclate RMSE on test dataset

# save the result
df_results.loc[poly_degree] = [R2, rmse_train, rmse_test]
#display(df_results)


# Drawing the  regression line
np_X0 = np.arange(1960, 2023, 1) # [1960, 1951, ... , 2022]
df_X0 = pd.DataFrame(data = np_X0, columns=["year"])

std_X0 = scaler.transform(df_X0[["year"]])      # Standardization for dammy input
X0 = pf.transform(std_X0)                       # extend X0 to polynomial
y0 = model.predict(X0)

plt.scatter(X_train[:,0], t_train, c="blue")
plt.plot(X0[:,0], y0, c="black")
plt.show()

plt.scatter(X_test[:,0], t_test, c="orange")
plt.plot(X0[:,0], y0, c="black")
plt.show()





#=============== Exercise 1.2.2 ============================
X = df_icecream[["temperature"]]            # extracting "temperature" column as input X
X["temperature^2"] = X["temperature"]**2    # adding "temperature^2" column to the input X


t = df_icecream[["ice cream sales"]]        # extracting 'ice cream sales' column as target t

from sklearn.linear_model import LinearRegression

# Definition of a linear regression model, optimization and prediction
model = LinearRegression(fit_intercept = True)  # If fit_intercept = True, the intercept b will be used
model.fit(X, t)         # optimization (obtaining a0, a1 and b)
y = model.predict(X)    # prediction

# Model evaluation
R2 = model.score(X,t)
rmse = np.sqrt(np.average((t-y)**2))

print("A2 =", model.coef_)       # This case, there are 2 parameters: a[0] and a[1]
print("b2 =", model.intercept_)
print("R^2_2 =", R2)
print("RMSE2=", rmse)

#==3==


X = df_icecream[["temperature"]]            # extracting "temperature" column as input X
X["temperature^2"] = X["temperature"]**2    # adding "temperature^2" column to the input X
X["temperature^3"] = X["temperature"]**3    # adding "temperature^2" column to the input X


t = df_icecream[["ice cream sales"]]        # extracting 'ice cream sales' column as target t

from sklearn.linear_model import LinearRegression

# Definition of a linear regression model, optimization and prediction
model = LinearRegression(fit_intercept = True)  # If fit_intercept = True, the intercept b will be used
model.fit(X, t)         # optimization (obtaining a0, a1 and b)
y = model.predict(X)    # prediction

# Model evaluation
R2 = model.score(X,t)
rmse = np.sqrt(np.average((t-y)**2))

print("A3 =", model.coef_)       # This case, there are 2 parameters: a[0] and a[1]
print("b3 =", model.intercept_)
print("R^2_3 =", R2)
print("RMSE3=", rmse)


#==4==


X = df_icecream[["temperature"]]            # extracting "temperature" column as input X
X["temperature^2"] = X["temperature"]**2    # adding "temperature^2" column to the input X
X["temperature^3"] = X["temperature"]**3    # adding "temperature^2" column to the input X
X["temperature^4"] = X["temperature"]**4    # adding "temperature^2" column to the input X


t = df_icecream[["ice cream sales"]]        # extracting 'ice cream sales' column as target t

from sklearn.linear_model import LinearRegression

# Definition of a linear regression model, optimization and prediction
model = LinearRegression(fit_intercept = True)  # If fit_intercept = True, the intercept b will be used
model.fit(X, t)         # optimization (obtaining a0, a1 and b)
y = model.predict(X)    # prediction

# Model evaluation
R2 = model.score(X,t)
rmse = np.sqrt(np.average((t-y)**2))

print("A4 =", model.coef_)       # This case, there are 2 parameters: a[0] and a[1]
print("b4 =", model.intercept_)
print("R^2_4 =", R2)
print("RMSE4=", rmse)


#==5==


X = df_icecream[["temperature"]]            # extracting "temperature" column as input X
X["temperature^2"] = X["temperature"]**2    # adding "temperature^2" column to the input X
X["temperature^3"] = X["temperature"]**3    # adding "temperature^2" column to the input X
X["temperature^4"] = X["temperature"]**4    # adding "temperature^2" column to the input X
X["temperature^5"] = X["temperature"]**5    # adding "temperature^2" column to the input X


t = df_icecream[["ice cream sales"]]        # extracting 'ice cream sales' column as target t

from sklearn.linear_model import LinearRegression

# Definition of a linear regression model, optimization and prediction
model = LinearRegression(fit_intercept = True)  # If fit_intercept = True, the intercept b will be used
model.fit(X, t)         # optimization (obtaining a0, a1 and b)
y = model.predict(X)    # prediction

# Model evaluation
R2 = model.score(X,t)
rmse = np.sqrt(np.average((t-y)**2))

print("A5 =", model.coef_)       # This case, there are 2 parameters: a[0] and a[1]
print("b5 =", model.intercept_)
print("R^2_5 =", R2)
print("RMSE5=", rmse)



#==6==


X = df_icecream[["temperature"]]            # extracting "temperature" column as input X
X["temperature^2"] = X["temperature"]**2    # adding "temperature^2" column to the input X
X["temperature^3"] = X["temperature"]**3    # adding "temperature^2" column to the input X
X["temperature^4"] = X["temperature"]**4    # adding "temperature^2" column to the input X
X["temperature^5"] = X["temperature"]**5    # adding "temperature^2" column to the input X
X["temperature^6"] = X["temperature"]**6    # adding "temperature^2" column to the input X


t = df_icecream[["ice cream sales"]]        # extracting 'ice cream sales' column as target t

from sklearn.linear_model import LinearRegression

# Definition of a linear regression model, optimization and prediction
model = LinearRegression(fit_intercept = True)  # If fit_intercept = True, the intercept b will be used
model.fit(X, t)         # optimization (obtaining a0, a1 and b)
y = model.predict(X)    # prediction

# Model evaluation
R2 = model.score(X,t)
rmse = np.sqrt(np.average((t-y)**2))

print("A6 =", model.coef_)       # This case, there are 2 parameters: a[0] and a[1]
print("b6 =", model.intercept_)
print("R^2_6 =", R2)
print("RMSE6=", rmse)

#==7==


X = df_icecream[["temperature"]]            # extracting "temperature" column as input X
X["temperature^2"] = X["temperature"]**2    # adding "temperature^2" column to the input X
X["temperature^3"] = X["temperature"]**3    # adding "temperature^2" column to the input X
X["temperature^4"] = X["temperature"]**4    # adding "temperature^2" column to the input X
X["temperature^5"] = X["temperature"]**5    # adding "temperature^2" column to the input X
X["temperature^6"] = X["temperature"]**6    # adding "temperature^2" column to the input X
X["temperature^7"] = X["temperature"]**7    # adding "temperature^2" column to the input X


t = df_icecream[["ice cream sales"]]        # extracting 'ice cream sales' column as target t

from sklearn.linear_model import LinearRegression

# Definition of a linear regression model, optimization and prediction
model = LinearRegression(fit_intercept = True)  # If fit_intercept = True, the intercept b will be used
model.fit(X, t)         # optimization (obtaining a0, a1 and b)
y = model.predict(X)    # prediction

# Model evaluation
R2 = model.score(X,t)
rmse = np.sqrt(np.average((t-y)**2))

print("A7 =", model.coef_)       # This case, there are 2 parameters: a[0] and a[1]
print("b7 =", model.intercept_)
print("R^2_7 =", R2)
print("RMSE7=", rmse)

#==8==

X = df_icecream[["temperature"]]            # extracting "temperature" column as input X
X["temperature^2"] = X["temperature"]**2    # adding "temperature^2" column to the input X
X["temperature^3"] = X["temperature"]**3    # adding "temperature^2" column to the input X
X["temperature^4"] = X["temperature"]**4    # adding "temperature^2" column to the input X
X["temperature^5"] = X["temperature"]**5    # adding "temperature^2" column to the input X
X["temperature^6"] = X["temperature"]**6    # adding "temperature^2" column to the input X
X["temperature^7"] = X["temperature"]**7    # adding "temperature^2" column to the input X
X["temperature^8"] = X["temperature"]**8    # adding "temperature^2" column to the input X

t = df_icecream[["ice cream sales"]]        # extracting 'ice cream sales' column as target t

from sklearn.linear_model import LinearRegression

# Definition of a linear regression model, optimization and prediction
model = LinearRegression(fit_intercept = True)  # If fit_intercept = True, the intercept b will be used
model.fit(X, t)         # optimization (obtaining a0, a1 and b)
y = model.predict(X)    # prediction

# Model evaluation
R2 = model.score(X,t)
rmse = np.sqrt(np.average((t-y)**2))

print("A8 =", model.coef_)       # This case, there are 2 parameters: a[0] and a[1]
print("b8 =", model.intercept_)
print("R^2_8 =", R2)
print("RMSE8=", rmse)


#==9==


X = df_icecream[["temperature"]]            # extracting "temperature" column as input X
X["temperature^2"] = X["temperature"]**2    # adding "temperature^2" column to the input X
X["temperature^3"] = X["temperature"]**3    # adding "temperature^2" column to the input X
X["temperature^4"] = X["temperature"]**4    # adding "temperature^2" column to the input X
X["temperature^5"] = X["temperature"]**5    # adding "temperature^2" column to the input X
X["temperature^6"] = X["temperature"]**6    # adding "temperature^2" column to the input X
X["temperature^7"] = X["temperature"]**7    # adding "temperature^2" column to the input X
X["temperature^8"] = X["temperature"]**8    # adding "temperature^2" column to the input X
X["temperature^9"] = X["temperature"]**9    # adding "temperature^2" column to the input X

t = df_icecream[["ice cream sales"]]        # extracting 'ice cream sales' column as target t

from sklearn.linear_model import LinearRegression

# Definition of a linear regression model, optimization and prediction
model = LinearRegression(fit_intercept = True)  # If fit_intercept = True, the intercept b will be used
model.fit(X, t)         # optimization (obtaining a0, a1 and b)
y = model.predict(X)    # prediction

# Model evaluation
R2 = model.score(X,t)
rmse = np.sqrt(np.average((t-y)**2))

print("A9 =", model.coef_)       # This case, there are 2 parameters: a[0] and a[1]
print("b9 =", model.intercept_)
print("R^2_9 =", R2)
print("RMSE9=", rmse)



#==10==


X = df_icecream[["temperature"]]            # extracting "temperature" column as input X
X["temperature^2"] = X["temperature"]**2    # adding "temperature^2" column to the input X
X["temperature^3"] = X["temperature"]**3    # adding "temperature^2" column to the input X
X["temperature^4"] = X["temperature"]**4    # adding "temperature^2" column to the input X
X["temperature^5"] = X["temperature"]**5    # adding "temperature^2" column to the input X
X["temperature^6"] = X["temperature"]**6    # adding "temperature^2" column to the input X
X["temperature^7"] = X["temperature"]**7    # adding "temperature^2" column to the input X
X["temperature^8"] = X["temperature"]**8    # adding "temperature^2" column to the input X
X["temperature^9"] = X["temperature"]**9    # adding "temperature^2" column to the input X
X["temperature^10"] = X["temperature"]**10    # adding "temperature^2" column to the input X

t = df_icecream[["ice cream sales"]]        # extracting 'ice cream sales' column as target t

from sklearn.linear_model import LinearRegression

# Definition of a linear regression model, optimization and prediction
model = LinearRegression(fit_intercept = True)  # If fit_intercept = True, the intercept b will be used
model.fit(X, t)         # optimization (obtaining a0, a1 and b)
y = model.predict(X)    # prediction

# Model evaluation
R2 = model.score(X,t)
rmse = np.sqrt(np.average((t-y)**2))

print("A10 =", model.coef_)       # This case, there are 2 parameters: a[0] and a[1]
print("b10 =", model.intercept_)
print("R^2_10 =", R2)
print("RMSE10=", rmse)



#============ Exercise 1.2.3 =====================
#In my opinion about extrapolation, the model might give an intuitive hint for a human being, but it can never be blindly trusted.
