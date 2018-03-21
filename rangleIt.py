#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 22:52:23 2017

@author: prakritibansal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm #import statsmodel


def drawLine(model, X_test, y_test, title, x_label, y_label, labels):
  # This convenience method will take care of plotting your
  # test observations, comparing them to the regression line,
  # and displaying the R2 coefficient
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(X_test, y_test, c='g', marker='o')
  ax.plot(X_test, model.predict(X_test), color='orange', linewidth=1, alpha=0.7)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.set_title(title)
  
  for i, l in enumerate(labels):                                       # <--
    ax.annotate(l, (X_test[i], y_test[i])) # <--

  print title
  print "Intercept(s): ", model.intercept_

  plt.show()
  

#-----------------LOADING DATA--------------------------------

lit = pd.read_csv('Datasets/Literacy.csv')
emp= pd.read_csv('Datasets/total unem.csv')
fac = pd.read_csv('Datasets/factory.csv')
ind = pd.read_csv('Datasets/indicators.csv')
corpt = pd.read_csv('Datasets/corruption.csv')




#-------------------- DATA WRANGLING ------------------------

ind["Population (in 000's) 2011 (2)"]= ind["Population (in 000's) 2011 (2)"].str.replace(',', '')
ind["Population (in 000's) 2011 (2)"] = pd.to_numeric(ind["Population (in 000's) 2011 (2)"])
ind["Annual Exponential Growth Rate(%) 2001-2011 (3)"][ind["Annual Exponential Growth Rate(%) 2001-2011 (3)"] == "(-)0.05"] = '-0.05'
ind["Annual Exponential Growth Rate(%) 2001-2011 (3)"] = pd.to_numeric(ind["Annual Exponential Growth Rate(%) 2001-2011 (3)"])
ind.drop(['Sl.No.'], axis = 1, inplace = True)
ind.drop([24, 35], axis = 0, inplace = True)
ind.sort_values(['State/UT (1)'], ascending = True, inplace = True)
ind.reset_index(drop = True, inplace = True)
#print ind.head()


lit.drop([23], axis = 0, inplace = True)
lit.reset_index(drop = True, inplace = True)

fac.drop(['State/Union Territory'], axis = 1, inplace = True)
fac.sort_values(['State/Union Territory Name'], inplace = True)
fac['Number of Factories - 2011-12'].fillna(fac['Number of Factories - 2011-12'].mean(),  
       inplace = True)
fac.reset_index(drop= True, inplace = True)
emp.drop([35, 18], axis = 0, inplace = True)
emp.sort_values(['State/UTs'], ascending = True, inplace = True)
emp.reset_index(drop = True, inplace = True)


#print corpt.dtypes
corpt = corpt[corpt.YEAR == 2011]

corpt.sort_values(["STATE/UT"], ascending = True, inplace = True)
corpt.drop([211], axis = 0, inplace = True)
corpt.reset_index(drop = True, inplace = True)
#print corpt
#print emp.head()

#print fac.head()

#print fac.describe()


#print lit.head()


#print emp.describe()
#print lit.describe()

#print emp.dtypes
#print lit.dtypes




l = corpt["STATE/UT"]
#print l
growth = ind['Annual Exponential Growth Rate(%) 2001-2011 (3)']
pop11 = ind["Population (in 000's) 2011 (2)"]
fac11 = fac['Number of Factories - 2011-12']
emp11 = emp["2011-12 - Rural+Urban"]
lit11 = lit["2011"]
genra = ind["Sex Ratio(Females per 1000 males) 2011 (5)"]
corpt11 = corpt["Total Amt. Of Fine Imposed During The Year (In Rupees)"]
X = pd.DataFrame({
        "State/UT": l,
        "Bias" : 1,
        "Literacy" : lit11,
        "Factories": fac11,
        "Population": pop11,
        "Annual Growth": growth,
        "Gender Ratio": genra,
        "Corruption" : corpt11,
        "Unemployment rate": emp11
        })
X.to_csv(path_or_buf = 'Datasets/finalData.csv', index = True)
print X.describe()
print "------------------------------------------------------------------"
print "------------------------------------------------------------------"
print X.corr()
correl = X.corr()
desc = X.describe()
correl.to_csv(path_or_buf = 'Datasets/correlation.csv', index = True)
desc.to_csv(path_or_buf = 'Datasets/description.csv', index = True)

plt.imshow(X.corr(), cmap = plt.cm.Reds, interpolation = 'nearest')
plt.colorbar()
tick_marks = [i for i in range (len(X.columns))]
plt.xticks(tick_marks, X.columns, rotation = 'vertical')
plt.yticks(tick_marks, X.columns)
plt.show()



corpt11 = corpt11.to_frame()
emp11 = emp11.to_frame()
lit11 = lit11.to_frame()
fac11 = fac11.to_frame()
pop11 = pop11.to_frame()
growth= growth.to_frame()
genra = genra.to_frame()
X.drop(["Unemployment rate", "State/UT"], axis = 1, inplace = True)
#print emp11




#---------------FEATURE SCALING -----------------

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
XS = scale.fit_transform(X)
emp11 = scale.fit_transform(emp11)
lit11 = scale.fit_transform(lit11)
fac11 = scale.fit_transform(fac11)
pop11= scale.fit_transform(pop11)
growth = scale.fit_transform(growth)
genra = scale.fit_transform(genra)
corpt11 = scale.fit_transform(corpt11)
XS = pd.DataFrame(XS,columns =X.columns )
#print X.describe()

#print emp11[24]
#print len(fac11), len(emp11), len(lit11)


#---------------STATISTICAL TESTING -----------------------

result= sm.OLS(emp11, X).fit()
print result.summary()


#----------LINEAR REGRESSION PLOTS -----------------------
#Uncomment to see the plots for each of the independent variables.

from sklearn.linear_model import LinearRegression

model= LinearRegression()
model.fit(lit11, emp11 )
score = model.score(lit11, emp11)
drawLine(model, lit11, emp11, "Literacy/Unemployment (2011)", "Literacy", "Unemployment", l)

model.fit(fac11, emp11 )
score = model.score(fac11, emp11)
#drawLine(model, fac11, emp11, "Factories/Unemployment (2011)", "Factories", "Unemployment", l)

model.fit(pop11, emp11 )
score = model.score(pop11, emp11)
#drawLine(model, pop11, emp11, "Population/Unemployment (2011)", "Population", "Unemployment", l)

model.fit(growth, emp11 )
score = model.score(growth, emp11)
#drawLine(model, growth, emp11, "Annual Growth rate /Unemployment (2011)", "Annual Growth Rate", "Unemployment", l)

model.fit(genra, emp11 )
score = model.score(genra, emp11)
#drawLine(model, genra, emp11, "Gender Ratio /Unemployment (2011)", "Gender Ratio", "Unemployment", l)

model.fit(corpt11, emp11 )
score = model.score(corpt11, emp11)
#drawLine(model, corpt11, emp11, "Corruption/Unemployment (2011)", "Corruption", "Unemployment", l)

model.fit(X, emp11)
score = model.score(X, emp11)
print score



#---------NOTES----------------------
# The final cleaned up data used in the project 
# can be viewed in /Datasets folder.
# Mizoram was deleted from the datasets because
# it's data was not given in one dataset. Therefore,
# the overall data points are 34 rather than 35.


