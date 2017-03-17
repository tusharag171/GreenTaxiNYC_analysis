#Run this file to generate output for all questions. From terminal do : python Final_Code.py

import pandas as pd
import datetime
from pandas_datareader import data, wb
import matplotlib.pyplot as plt
import numpy as np
import math as m
from scipy.stats import ttest_ind
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.feature_selection import SelectFromModel
from xgboost import plot_importance
from matplotlib import pyplot 
import urllib
import csv
from sklearn import cross_validation, metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
import pickle

# Question 1 - a) Programmatically download and load into your favorite analytical tool the trip data for September 2015.
url = 'https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2015-09.csv'

print "Downloading data..."
urllib.urlretrieve(url, "data.csv")

df = pd.read_csv('data.csv')


# Question 1 - b) Report how many rows and columns of data you have loaded.

print 'Columns: ' +str(len(df.columns))
print 'Rows: '+str(len(df))


# Question 2 - a) Plot a histogram of the number of the trip distance ("Trip Distance").
#              b)  Report any structure you find and any hypotheses you have about that structure.

start = m.floor(min(df['Trip_distance']))
end =  m.ceil(max(df['Trip_distance']))

bins = np.arange(start,end,0.5)
plt.hist(df['Trip_distance'], bins, histtype='bar')
plt.xlabel('Trip distance (miles)')
plt.ylabel('Count of trips')
plt.title('Trip distance with outliers')
print 'Close the Image to continue...'
plt.show()


bins = np.arange(start,20,0.5)
plt.hist(df['Trip_distance'], bins, histtype='bar')
plt.xlabel('Trip distance (miles)')
plt.ylabel('Count of trips')
plt.title('Trip distance without outliers')
print 'Close the Image to continue...'
plt.show()



#Question 3 -a) Report mean and median trip distance grouped by hour of day.
#Question 3 -b) We'd like to get a rough sense of identifying trips that originate or terminate at one of the NYC area airports. 
#Can you provide a count of how many transactions fit this criteria, the average fair, and any other interesting characteristics of these trips.

df['Pickup_DateTime'] = df.lpep_pickup_datetime.apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
df['Dropoff_DateTime'] = df.Lpep_dropoff_datetime.apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
df['Pickup_Hour'] = df.Pickup_DateTime.apply(lambda x:x.hour)

Values_table = df.pivot_table(index='Pickup_Hour', values='Trip_distance',aggfunc=('mean','median')).reset_index()
Values_table.columns = ['Hour','Mean_Distance','Median_Distance']
print Values_table

airport_trips = df[ (df.RateCodeID==3) | (df.RateCodeID==2)]
print "Total num of trips to/from ", airport_trips.shape[0]
print "Average fare of trips to/from airports: $", airport_trips.Fare_amount.mean(),"per trip"
print "Average total fare amount before tipping of trips to/from NYC airports: $", airport_trips.Total_amount.mean(),"per trip"




# Question 4 -a) Build a derived variable for tip as a percentage of the total fare.

df = df[(df.Total_amount>=0)] #Cleaning Data; Note:- There could be a minimum value other that 0 for green taxis
df['Tip_Percentage'] = 100*df.Tip_amount/df.Total_amount

# Question 4 -b)  Build a predictive model for tip as a percentage of the total fare. 
#Use as much of the data as you like (or all of it). We will validate a sample.

# Creating some more derived variables

df['Trip_duration'] = ((df.Dropoff_DateTime-df.Pickup_DateTime).apply(lambda x:x.total_seconds()/60))
df['Avg_Speed_mph'] = df.Trip_distance/(df.Trip_duration/60)


ind =  df[(df.Avg_Speed_mph.notnull()) & (df.Avg_Speed_mph<240)].index
df2 = df.loc[ind].reset_index()
first_week = datetime.datetime(2015,9,1).isocalendar()[1] 
df2['Week_NUM'] = df2.Pickup_DateTime.apply(lambda x:x.isocalendar()[1])-first_week + 1

#Removing Anomalies / Cleaning DATA
df2['Trip_type '] = df2['Trip_type '].replace(np.NaN,1) # replacing with 1 as it is most frequent

#Cleaning RateCodeId
df2['RateCodeID'].value_counts()
indices_ri = df2[~((df2.RateCodeID>=1) & (df2.RateCodeID<=6))].index
df2.loc[indices_ri, 'RateCodeID'] = 1

#Removing Negatives if any
df2.Total_amount = df2.Total_amount.abs()
df2.Fare_amount = df2.Fare_amount.abs()
df2.improvement_surcharge = df2.improvement_surcharge.abs()
df2.Tip_amount = df2.Tip_amount.abs()
df2.Tolls_amount = df2.Tolls_amount.abs()
df2.MTA_tax = df2.MTA_tax.abs()

#Making extras zero for invalid values
ind = df2[~((df2.Extra==0) | (df2.Extra==0.5) | (df2.Extra==1))].index
df2.loc[ind, 'Extra'] = 0


no_tip =  len(df2[df2.Tip_amount==0])
tip =  len(df2[df2.Tip_amount>0])

df2['paid_tips'] = (df2.Tip_Percentage>0)

# Finding percentage of people not paying tips
print str((float)(no_tip)/(tip+no_tip) * 100) + ' percent people gave no tips '


train = df2.copy()
from sklearn.model_selection import KFold
kfold_5 = KFold(n_splits=5)
avg_mean_sq_val = avg_r2_score_val = 0

# Running 5-fold Cross Validation on complete dataset using RandomForestRegressor
train = df2.copy()
from sklearn.model_selection import KFold
kfold_5 = KFold(n_splits=5)
avg_mean_sq_val = 0

# Running 5-fold Cross Validation on complete dataset using RandomForestRegressor
for train_id, test_id in kfold_5.split(df2):
    train = train.loc[train_id]
    test = df2.loc[test_id]
    train['ID'] = train.index
    target = 'Tip_Percentage'
    p = ['Total_amount', 'Trip_duration', 'Avg_Speed_mph']
    train[train.Tip_Percentage.isnull()] = 0
    test[test.Tip_Percentage.isnull()] = 0
    rfr = RandomForestRegressor(n_estimators=200)
    model = rfr.fit(train[p],train[target])
    ypred = model.predict(test[p])
    mean_sq_val = metrics.mean_squared_error(ypred,test.Tip_Percentage)
    avg_mean_sq_val += mean_sq_val

    
print 'RandomForestRegressor test Avg mse:',float(avg_mean_sq_val/5)

#final model
model_f = rfr.fit(train[p],train[target])
with open('classifier.pkl','wb') as f_id:
    pickle.dump(model_f,f_id)
    f_id.close()


# Question 5 - Option A: Distributions
# a) Build a derived variable representing the average speed over the course of a trip.
# b)Can you perform a test to determine if the average trip speeds are materially the same in all weeks of September? 
#If you decide they are not the same, can you form a hypothesis regarding why they differ?
#c) Can you build up a hypothesis of average trip speed as a function of time of day?

print 'Avergage Speed over the course has been already calculated and stored under Avg_Speed_mph'
print "mean speed by week:\n", df2[['Avg_Speed_mph','Week_NUM']].groupby('Week_NUM').mean()


weeks = pd.unique(df2.Week_NUM)
pvals = []

# Performing Paired ttest. Explaination in document
for i in range(len(weeks)): 
    for j in range(len(weeks)):
        pvals.append((weeks[i], weeks[j],ttest_ind(df2[df2.Week_NUM==weeks[i]].Avg_Speed_mph,df2[df2.Week_NUM==weeks[j]].Avg_Speed_mph,False)[1]))

pvalues = pd.DataFrame(pvals,columns=['week1','week2','pval'])
print "p-values are:\n",pvalues.pivot_table(index='week1',columns='week2',values='pval')


hours = range(24)
df2['Hour'] = df2.Pickup_DateTime.apply(lambda x:x.hour)
# boxplot
df2.boxplot('Avg_Speed_mph','Hour')
plt.ylim([5,24]) # cut off outliers
plt.ylabel('Speed (mph)')
plt.show()


