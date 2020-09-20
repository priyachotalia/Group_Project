import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

rent_df=pd.read_csv("data/Rent_Data.csv")
rent_df.columns = ["Year", "Quarter", "LGA", "Dwelling Type", "Bedroom Number", 
                "First QNB", "Second QNB", "Third QNB", "NB Lodged", "TB Lodged", "Qtrly Median Change", 
                "Annly Median Change", "Qtrly NB", "Annly NB"]

### DATA CLEANING ###
rent_df = rent_df.replace("-", np.nan)
rent_df["First QNB"] = rent_df["First QNB"].astype(str).astype(float)
rent_df["Second QNB"] = rent_df["Second QNB"].astype(str).astype(float)
rent_df["Third QNB"] = rent_df["Third QNB"].astype(str).astype(float)

## NAN values are replaced with min-1
f_min_val = np.min(rent_df["First QNB"]) - 1
s_min_val = np.min(rent_df["Second QNB"]) - 1
t_min_val = np.min(rent_df["Third QNB"]) - 1

null = []
for x in rent_df.index:#range(len(rent_df.index)):
    null_row = rent_df.loc[x].isnull().sum()
    if null_row == 9:
        null.append(x)
        rent_df.loc[x] = rent_df.loc[x].replace(np.nan, 0)

## NAN values are replaced with min-1 

rent_df["First QNB"] = rent_df["First QNB"].replace(np.nan, f_min_val)
rent_df["Second QNB"] = rent_df["Second QNB"].replace(np.nan, s_min_val)
rent_df["Third QNB"] = rent_df["Third QNB"].replace(np.nan, t_min_val)

## replace s  with 1 
rent_df = rent_df.replace("s", 1)
rent_df["NB Lodged"] = rent_df["NB Lodged"].astype(str).astype(float)
rent_df["TB Lodged"] = rent_df["TB Lodged"].astype(str).astype(float)

### NAN values are replaced with 0 ###
## s values which mean really small values are replaced with min/2

fq_min_val = np.min(rent_df["NB Lodged"])
rent_df["NB Lodged"] = rent_df["NB Lodged"].replace(1, fq_min_val)/2
rent_df["NB Lodged"] = rent_df["NB Lodged"].replace(np.nan, 0)

sq_min_val = np.min(rent_df["TB Lodged"])
rent_df["TB Lodged"] = rent_df["TB Lodged"].replace(1, sq_min_val)/2
rent_df["TB Lodged"] = rent_df["TB Lodged"].replace(np.nan, 0)

## float to int
rent_df["NB Lodged"] = rent_df["NB Lodged"].astype(int)
rent_df["TB Lodged"] = rent_df["TB Lodged"].astype(int)

rent_df["First QNB"] = rent_df["First QNB"].astype(int)
rent_df["Second QNB"] = rent_df["Second QNB"].astype(int)
rent_df["Third QNB"] = rent_df["Third QNB"].astype(int)

## remove % sign 
rent_df['Qtrly Median Change'] = rent_df['Qtrly Median Change'].astype(str).str.extract('(\d+)').astype(float)
rent_df['Annly Median Change'] = rent_df['Annly Median Change'].astype(str).str.extract('(\d+)').astype(float)
rent_df['Qtrly NB'] = rent_df['Qtrly NB'].astype(str).str.extract('(\d+)').astype(float)
rent_df['Annly NB'] = rent_df['Annly NB'].astype(str).str.extract('(\d+)').astype(float)

## replace nan values for the last 4 columns with 0 
rent_df = rent_df.replace(np.nan, 0)

price_df = rent_df[['LGA', 'Dwelling Type', 'Bedroom Number', 'First QNB', 'Second QNB', 'Third QNB']].copy()
price_df['mean'] = price_df.mean(axis=1)

rent_df['Mean QNB'] = np.nan
rent_df['Mean QNB'] = price_df['mean']
rent_df['Mean QNB'] = rent_df['Mean QNB'].astype(int)

### GET TOP LGA ###
## remove Total column from LGA, and get unique LGA values 
sort_df = rent_df[rent_df['LGA']!='Total']
top_mean = sort_df.sort_values(by="Mean QNB", ascending = False)['LGA'].unique()[:10]
top_mean_arr = []

## Get the Mean QNB per quarter and year of top LGA for graphing 
for x in range(len(top_mean)):
    name = top_mean[x]
    top_mean_df = rent_df[rent_df['LGA']==top_mean[x]].sort_values(by="Mean QNB", ascending = False)
    top_mean_df = top_mean_df[top_mean_df['Dwelling Type'] == 'Total']
    top_mean_df = top_mean_df[top_mean_df['Bedroom Number'] == 'Total']
    top_mean_df = top_mean_df.sort_values(by=['Year', 'Quarter']) 
    top_mean_df = top_mean_df.reset_index()
    for y in top_mean_df.index:
        if y == 0:
            Q1_15 = top_mean_df['Mean QNB'].loc[y]
        if y == 1:
            Q2_15 = top_mean_df['Mean QNB'].loc[y]
        if y == 2:
            Q3_15 = top_mean_df['Mean QNB'].loc[y]
        if y == 3:
            Q4_15 = top_mean_df['Mean QNB'].loc[y]
        if y == 4:
            Q1_16 = top_mean_df['Mean QNB'].loc[y]
        if y == 5:
            Q2_16 = top_mean_df['Mean QNB'].loc[y]
        if y == 6:
            Q3_16 = top_mean_df['Mean QNB'].loc[y]
        if y == 7:
            Q4_16 = top_mean_df['Mean QNB'].loc[y]
        if y == 8:
            Q1_17 = top_mean_df['Mean QNB'].loc[y]
        if y == 9:
            Q2_17 = top_mean_df['Mean QNB'].loc[y]
        if y == 10:
            Q3_17 = top_mean_df['Mean QNB'].loc[y]
        if y == 11:
            Q4_17 = top_mean_df['Mean QNB'].loc[y]
        if y == 12:
            Q1_18 = top_mean_df['Mean QNB'].loc[y]
        if y == 13:
            Q2_18 = top_mean_df['Mean QNB'].loc[y]
        if y == 14:
            Q3_18 = top_mean_df['Mean QNB'].loc[y]
        if y == 15:
            Q4_18 = top_mean_df['Mean QNB'].loc[y]
        if y == 16:
            Q1_19 = top_mean_df['Mean QNB'].loc[y]
        if y == 17:
            Q2_19 = top_mean_df['Mean QNB'].loc[y]
    top_mean_arr.append((name, Q1_15, Q2_15, Q3_15, Q4_15, Q1_16, Q2_16, Q3_16, Q4_16, 
                    Q1_17, Q2_17, Q3_17, Q4_17, Q1_18, Q2_18, Q3_18, Q4_18,
                    Q1_19, Q2_19))

## Graph LGA
top_mean_df = pd.DataFrame.from_records(top_mean_arr)
top_mean_df.columns = ['LGA', '2015.Q1', '2015.Q2', '2015.Q3', '2015.Q4', '2016.Q1', '2016.Q2', '2016.Q3', '2016.Q4',
                      '2017.Q1', '2017.Q2', '2017.Q3', '2017.Q4', '2018.Q1', '2018.Q2', '2018.Q3', '2018.Q4',
                      '2019.Q1', '2019.Q2']
top_mean_df.index = top_mean_df['LGA']
top_mean_df = top_mean_df.drop('LGA', axis=1)

## remove columns for the graph to look nice
top_mean_df = top_mean_df.drop(['2015.Q1', '2015.Q2', '2015.Q3', '2015.Q4', '2016.Q1', '2016.Q2', '2016.Q3', '2016.Q4',
                      '2017.Q1', '2017.Q2'], axis=1)

### Get all LGAs and Mean QNB per year and quarter ###
## Get unique LGA
LGA_list = sort_df['LGA'].unique()
all_mean_arr = []

## Get LGA renting price per year, append in array 
for x in range(len(LGA_list)):
    name = LGA_list[x]
    all_mean_df = rent_df[rent_df['LGA']==LGA_list[x]].sort_values(by="Mean QNB", ascending = False)
    all_mean_df = all_mean_df[all_mean_df['Dwelling Type'] == 'Total']
    all_mean_df = all_mean_df[all_mean_df['Bedroom Number'] == 'Total']
    all_mean_df = all_mean_df.sort_values(by=['Year', 'Quarter']) 
    all_mean_df = all_mean_df.reset_index()
    for y in all_mean_df.index:
        if y == 0:
            Q1_15 = all_mean_df['Mean QNB'].loc[y]
        if y == 1:
            Q2_15 = all_mean_df['Mean QNB'].loc[y]
        if y == 2:
            Q3_15 = all_mean_df['Mean QNB'].loc[y]
        if y == 3:
            Q4_15 = all_mean_df['Mean QNB'].loc[y]
        if y == 4:
            Q1_16 = all_mean_df['Mean QNB'].loc[y]
        if y == 5:
            Q2_16 = all_mean_df['Mean QNB'].loc[y]
        if y == 6:
            Q3_16 = all_mean_df['Mean QNB'].loc[y]
        if y == 7:
            Q4_16 = all_mean_df['Mean QNB'].loc[y]
        if y == 8:
            Q1_17 = all_mean_df['Mean QNB'].loc[y]
        if y == 9:
            Q2_17 = all_mean_df['Mean QNB'].loc[y]
        if y == 10:
            Q3_17 = all_mean_df['Mean QNB'].loc[y]
        if y == 11:
            Q4_17 = all_mean_df['Mean QNB'].loc[y]
        if y == 12:
            Q1_18 = all_mean_df['Mean QNB'].loc[y]
        if y == 13:
            Q2_18 = all_mean_df['Mean QNB'].loc[y]
        if y == 14:
            Q3_18 = all_mean_df['Mean QNB'].loc[y]
        if y == 15:
            Q4_18 = all_mean_df['Mean QNB'].loc[y]
        if y == 16:
            Q1_19 = all_mean_df['Mean QNB'].loc[y]
        if y == 17:
            Q2_19 = all_mean_df['Mean QNB'].loc[y]
    all_mean_arr.append((name, Q1_15, Q2_15, Q3_15, Q4_15, Q1_16, Q2_16, Q3_16, Q4_16, 
                    Q1_17, Q2_17, Q3_17, Q4_17, Q1_18, Q2_18, Q3_18, Q4_18,
                    Q1_19, Q2_19))
## Array to dataframe
all_mean_df = pd.DataFrame.from_records(all_mean_arr)
all_mean_df.columns = ['LGA', '2015.Q1', '2015.Q2', '2015.Q3', '2015.Q4', '2016.Q1', '2016.Q2', '2016.Q3', '2016.Q4',
                      '2017.Q1', '2017.Q2', '2017.Q3', '2017.Q4', '2018.Q1', '2018.Q2', '2018.Q3', '2018.Q4',
                      '2019.Q1', '2019.Q2']
all_mean_df.index = all_mean_df['LGA']
#all_LGA_df = all_LGA_df.drop('LGA', axis=1)

## remove columns for the graph to look nice ###
all_mean_df = all_mean_df.drop(['2015.Q1', '2015.Q2', '2015.Q3', '2015.Q4', '2016.Q1', '2016.Q2', '2016.Q3', '2016.Q4',
                      '2017.Q1', '2017.Q2'], axis=1)


### TRAIN TEST for all_mean_df dataframe ###
## divide data
from sklearn.model_selection import train_test_split

cols = np.array(['2017.Q3', '2017.Q4', '2018.Q1', '2018.Q2', '2018.Q3', '2018.Q4',
                      '2019.Q1'])
X = all_mean_df[cols]
y = all_mean_df['2019.Q2']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.3)

## Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import mean_squared_error, r2_score

model_list = [DecisionTreeClassifier(random_state=0),
              RandomForestRegressor(random_state=1),
              svm.SVC(gamma='scale'),
              MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=20, alpha=0.01, 
                            solver='sgd', verbose=10,  random_state=21,tol=0.000000001)]
model_name = ['Decision Tree', 'Random Forest', 'SVM', 'Neural Network']

stats = []

for x in range(4):
    model = model_list[x]
    model.fit(X_train, y_train)
    predicted = model.predict(X_train)
    MSE_train = mean_squared_error(y_train, predicted)
    R2_train = r2_score(y_train, predicted)
    Accuracy_train = model.score(X_train, y_train)
    predicted = model.predict(X_test)
    MSE_test = mean_squared_error(y_test, predicted)
    R2_test = r2_score(y_test, predicted)
    Accuracy_test = model.score(X_test, y_test)
    model = model_name[x]
    stats.append((model, MSE_train, MSE_test, R2_train, R2_test, Accuracy_train, Accuracy_test))
    stats_df = pd.DataFrame.from_records(stats)
    stats_df.columns = ['Model', 'RMSE Train', 'RMSE Test', 'R-Squared Train', 'R-Squared Test', 
                        'Accuracy Score Train', 'Accuracy Score Test']
    stats_df.index = stats_df['Model']
    stats_df = stats_df.drop('Model',
                axis = 1)

### Top LGAs according to New Signed Bonds ###
## remove Total column from LGA, and get unique LGA values 
sort_df = rent_df[rent_df['LGA']!='Total']
top_NB = sort_df.sort_values(by="NB Lodged", ascending = False)['LGA'].unique()[:10]
top_NB_arr = []

## Get the Mean QNB per quarter and year of top LGA for graphing 
for x in range(len(top_NB)):
    name = top_NB[x]
    top_NB_df = rent_df[rent_df['LGA']==top_NB[x]].sort_values(by="NB Lodged", ascending = False)
    top_NB_df = top_NB_df[top_NB_df['Dwelling Type'] == 'Total']
    top_NB_df = top_NB_df[top_NB_df['Bedroom Number'] == 'Total']
    top_NB_df = top_NB_df.sort_values(by=['Year', 'Quarter']) 
    top_NB_df = top_NB_df.reset_index()
    for y in top_NB_df.index:
        if y == 0:
            Q1_15 = top_NB_df['NB Lodged'].loc[y]
        if y == 1:
            Q2_15 = top_NB_df['NB Lodged'].loc[y]
        if y == 2:
            Q3_15 = top_NB_df['NB Lodged'].loc[y]
        if y == 3:
            Q4_15 = top_NB_df['NB Lodged'].loc[y]
        if y == 4:
            Q1_16 = top_NB_df['NB Lodged'].loc[y]
        if y == 5:
            Q2_16 = top_NB_df['NB Lodged'].loc[y]
        if y == 6:
            Q3_16 = top_NB_df['NB Lodged'].loc[y]
        if y == 7:
            Q4_16 = top_NB_df['NB Lodged'].loc[y]
        if y == 8:
            Q1_17 = top_NB_df['NB Lodged'].loc[y]
        if y == 9:
            Q2_17 = top_NB_df['NB Lodged'].loc[y]
        if y == 10:
            Q3_17 = top_NB_df['NB Lodged'].loc[y]
        if y == 11:
            Q4_17 = top_NB_df['NB Lodged'].loc[y]
        if y == 12:
            Q1_18 = top_NB_df['NB Lodged'].loc[y]
        if y == 13:
            Q2_18 = top_NB_df['NB Lodged'].loc[y]
        if y == 14:
            Q3_18 = top_NB_df['NB Lodged'].loc[y]
        if y == 15:
            Q4_18 = top_NB_df['NB Lodged'].loc[y]
        if y == 16:
            Q1_19 = top_NB_df['NB Lodged'].loc[y]
        if y == 17:
            Q2_19 = top_NB_df['NB Lodged'].loc[y]
    top_NB_arr.append((name, Q1_15, Q2_15, Q3_15, Q4_15, Q1_16, Q2_16, Q3_16, Q4_16, 
                    Q1_17, Q2_17, Q3_17, Q4_17, Q1_18, Q2_18, Q3_18, Q4_18,
                    Q1_19, Q2_19))

## Get the Mean QNB per quarter and year of top LGA for graphing
top_nbmean_arr=[]

for x in range(len(top_NB)):
    name = top_NB[x]
    top_nbmean_df = rent_df[rent_df['LGA']==top_NB[x]].sort_values(by="Mean QNB", ascending = False)
    top_nbmean_df = top_nbmean_df[top_nbmean_df['Dwelling Type'] == 'Total']
    top_nbmean_df = top_nbmean_df[top_nbmean_df['Bedroom Number'] == 'Total']
    top_nbmean_df = top_nbmean_df.sort_values(by=['Year', 'Quarter']) 
    top_nbmean_df = top_nbmean_df.reset_index()
    for y in top_nbmean_df.index:
        if y == 0:
            Q1_15 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 1:
            Q2_15 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 2:
            Q3_15 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 3:
            Q4_15 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 4:
            Q1_16 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 5:
            Q2_16 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 6:
            Q3_16 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 7:
            Q4_16 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 8:
            Q1_17 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 9:
            Q2_17 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 10:
            Q3_17 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 11:
            Q4_17 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 12:
            Q1_18 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 13:
            Q2_18 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 14:
            Q3_18 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 15:
            Q4_18 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 16:
            Q1_19 = top_nbmean_df['Mean QNB'].loc[y]
        if y == 17:
            Q2_19 = top_nbmean_df['Mean QNB'].loc[y]
    top_nbmean_arr.append((name, Q1_15, Q2_15, Q3_15, Q4_15, Q1_16, Q2_16, Q3_16, Q4_16, 
                    Q1_17, Q2_17, Q3_17, Q4_17, Q1_18, Q2_18, Q3_18, Q4_18,
                    Q1_19, Q2_19))

## Graph LGA
top_NB_df = pd.DataFrame.from_records(top_NB_arr)
top_NB_df.columns = ['LGA', '2015.Q1', '2015.Q2', '2015.Q3', '2015.Q4', '2016.Q1', '2016.Q2', '2016.Q3', '2016.Q4',
                      '2017.Q1', '2017.Q2', '2017.Q3', '2017.Q4', '2018.Q1', '2018.Q2', '2018.Q3', '2018.Q4',
                      '2019.Q1', '2019.Q2']
top_NB_df.index = top_NB_df['LGA']
top_NB_df = top_NB_df.drop('LGA', axis=1)

## remove columns for the graph to look nice 
top_NB_df = top_NB_df.drop(['2015.Q1', '2015.Q2', '2015.Q3', '2015.Q4', '2016.Q1', '2016.Q2', '2016.Q3', '2016.Q4',
                      '2017.Q1', '2017.Q2'], axis=1)
## MEAN 
## Graph LGA
top_nbmean_df = pd.DataFrame.from_records(top_nbmean_arr)
top_nbmean_df.columns = ['LGA', '2015.Q1', '2015.Q2', '2015.Q3', '2015.Q4', '2016.Q1', '2016.Q2', '2016.Q3', '2016.Q4',
                      '2017.Q1', '2017.Q2', '2017.Q3', '2017.Q4', '2018.Q1', '2018.Q2', '2018.Q3', '2018.Q4',
                      '2019.Q1', '2019.Q2']
top_nbmean_df.index = top_nbmean_df['LGA']
top_nbmean_df = top_nbmean_df.drop('LGA', axis=1)

## remove columns for the graph to look nice
top_nbmean_df = top_nbmean_df.drop(['2015.Q1', '2015.Q2', '2015.Q3', '2015.Q4', '2016.Q1', '2016.Q2', '2016.Q3', '2016.Q4',
                      '2017.Q1', '2017.Q2'], axis=1)

### Dwelling Type Most People Rent ###
## dataframe per dwelling type
house_df = rent_df[rent_df['Dwelling Type']=='House']
flat_df = rent_df[rent_df['Dwelling Type']=='Flat/Unit']
townhouse_df = rent_df[rent_df['Dwelling Type']=='Townhouse']

## get total count per year
year = [2015, 2016, 2017, 2018, 2019]
bond_count_arr = []
for x in range(len(year)):
    yr = year[x]
    house = house_df[house_df['Year']==yr]['TB Lodged'].sum()
    name = 'House'
    bond_count_arr.append((yr, house, name))
    flat = flat_df[flat_df['Year']==yr]['TB Lodged'].sum()
    name = 'Flat/Unit'
    bond_count_arr.append((yr, flat, name))
    townhouse = townhouse_df[townhouse_df['Year']==yr]['TB Lodged'].sum()
    name = 'Townhouse'
    bond_count_arr.append((yr, townhouse, name))
    
### FLAT: number of bedrooms most people rent ###
bond_count_df = pd.DataFrame.from_records(bond_count_arr)
bond_count_df.columns = ['Year', 'Count', 'Dwelling Type']

BS_df = flat_df[flat_df['Bedroom Number']=='Bedsitter']
B1_df = flat_df[flat_df['Bedroom Number']=='1 Bedroom']
B2_df = flat_df[flat_df['Bedroom Number']=='2 Bedrooms']
B3_df = flat_df[flat_df['Bedroom Number']=='3 Bedrooms']
B4_df = flat_df[flat_df['Bedroom Number']=='4 or more Bedrooms']

flatbond_count_arr = []
for x in range(len(year)):
    yr = year[x]
    BS = BS_df[BS_df['Year']==yr]['TB Lodged'].sum()
    name = 'Bedsitter'
    flatbond_count_arr.append((yr, BS, name))
    B1 = B1_df[B1_df['Year']==yr]['TB Lodged'].sum()
    name = '1 Bedroom'
    flatbond_count_arr.append((yr, B1, name))
    B2 = B2_df[B2_df['Year']==yr]['TB Lodged'].sum()
    name = '2 Bedrooms'
    flatbond_count_arr.append((yr, B2, name))
    B3 = B3_df[B3_df['Year']==yr]['TB Lodged'].sum()
    name = '3 Bedrooms'
    flatbond_count_arr.append((yr, B3, name))
    B4 = B4_df[B4_df['Year']==yr]['TB Lodged'].sum()
    name = '4 or more Bedrooms'
    flatbond_count_arr.append((yr, B4, name))
    
flatbond_count_df = pd.DataFrame.from_records(flatbond_count_arr)
flatbond_count_df.columns = ['Year', 'Count', 'Number of Bedrooms']