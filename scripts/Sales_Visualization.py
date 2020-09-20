import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Import Data ###
sales_df=pd.read_csv("data/Sales_Data.csv")
sales_df.columns = ["Year", "Quarter", "LGA", "Dwelling Type", "First Quartile", "Second Quartile", 
                   "Third Quartile", "Mean Sales Price", "Sales Number", "Qtrly Median Change", 
                "Annly Median Change", "Qtrly Count Change", "Annly Count Change"]

### Clean Data ###
sales_df = sales_df.replace("-", np.nan)
sales_df = sales_df.replace("s",np.nan)
sales_df= sales_df.fillna(method='bfill')

sales_df['Year'] = sales_df['Year'].astype(int)
sales_df['Quarter'] = sales_df['Quarter'].astype(int)
sales_df["First Quartile"] = sales_df["First Quartile"].astype(str).astype(float)
sales_df['First Quartile'] = sales_df['First Quartile'].astype(int)
sales_df["Second Quartile"] = sales_df["Second Quartile"].astype(str).astype(float)
sales_df['Second Quartile'] = sales_df['Second Quartile'].astype(int)
sales_df["Third Quartile"] = sales_df["Third Quartile"].astype(str).astype(float)
sales_df['Third Quartile'] = sales_df['Third Quartile'].astype(int)
sales_df["Mean Sales Price"] = sales_df["Mean Sales Price"].astype(str).astype(float)
sales_df['Mean Sales Price'] = sales_df['Mean Sales Price'].astype(int)
sales_df["Sales Number"] = sales_df["Sales Number"].astype(str).astype(float)
sales_df['Sales Number'] = sales_df['Sales Number'].astype(int)

### GET TOP LGA ###
## remove Total column from LGA, and get unique LGA values 
sort_df = sales_df[sales_df['LGA']!='Total']
top_mean = sort_df.sort_values(by="Mean Sales Price", ascending = False)['LGA'].unique()[:10]
top_mean_arr = []

## Get the Mean Sales Price per quarter and year of top LGA for graphing 
for x in range(len(top_mean)):
    name = top_mean[x]
    top_mean_df = sales_df[sales_df['LGA']==top_mean[x]].sort_values(by="Mean Sales Price", ascending = False)
    top_mean_df = top_mean_df[top_mean_df['Dwelling Type'] == 'Total']
    top_mean_df = top_mean_df.sort_values(by=['Year', 'Quarter']) 
    top_mean_df = top_mean_df.reset_index()
    for y in top_mean_df.index:
        if y == 0:
            Q1_15 = top_mean_df['Mean Sales Price'].loc[y]
        if y == 1:
            Q2_15 = top_mean_df['Mean Sales Price'].loc[y]
        if y == 2:
            Q3_15 = top_mean_df['Mean Sales Price'].loc[y]
        if y == 3:
            Q4_15 = top_mean_df['Mean Sales Price'].loc[y]
        if y == 4:
            Q1_16 = top_mean_df['Mean Sales Price'].loc[y]
        if y == 5:
            Q2_16 = top_mean_df['Mean Sales Price'].loc[y]
        if y == 6:
            Q3_16 = top_mean_df['Mean Sales Price'].loc[y]
        if y == 7:
            Q4_16 = top_mean_df['Mean Sales Price'].loc[y]
        if y == 8:
            Q1_17 = top_mean_df['Mean Sales Price'].loc[y]
        if y == 9:
            Q2_17 = top_mean_df['Mean Sales Price'].loc[y]
        if y == 10:
            Q3_17 = top_mean_df['Mean Sales Price'].loc[y]
        if y == 11:
            Q4_17 = top_mean_df['Mean Sales Price'].loc[y]
        if y == 12:
            Q1_18 = top_mean_df['Mean Sales Price'].loc[y]
        if y == 13:
            Q2_18 = top_mean_df['Mean Sales Price'].loc[y]
        if y == 14:
            Q3_18 = top_mean_df['Mean Sales Price'].loc[y]
        if y == 15:
            Q4_18 = top_mean_df['Mean Sales Price'].loc[y]
    top_mean_arr.append((name, Q1_15, Q2_15, Q3_15, Q4_15, Q1_16, Q2_16, Q3_16, Q4_16, 
                    Q1_17, Q2_17, Q3_17, Q4_17, Q1_18, Q2_18, Q3_18, Q4_18))

## Graph LGA
top_mean_df = pd.DataFrame.from_records(top_mean_arr)
top_mean_df.columns = ['LGA', '2015.Q1', '2015.Q2', '2015.Q3', '2015.Q4', '2016.Q1', '2016.Q2', '2016.Q3', '2016.Q4',
                      '2017.Q1', '2017.Q2', '2017.Q3', '2017.Q4', '2018.Q1', '2018.Q2', '2018.Q3', '2018.Q4']
top_mean_df.index = top_mean_df['LGA']
top_mean_df = top_mean_df.drop('LGA', axis=1)

### Top LGAs according to New Signed Bonds ###
## remove Total column from LGA, and get unique LGA values 
sort_df = sales_df[sales_df['LGA']!='Total']
top_NB = sort_df.sort_values(by="Sales Number", ascending = False)['LGA'].unique()[:10]
top_NB_arr = []

## Get the Number of Sales per quarter and year of top LGA for graphing 
for x in range(len(top_NB)):
    name = top_NB[x]
    top_NB_df = sales_df[sales_df['LGA']==top_NB[x]].sort_values(by="Sales Number", ascending = False)
    top_NB_df = top_NB_df[top_NB_df['Dwelling Type'] == 'Total']
    top_NB_df = top_NB_df.sort_values(by=['Year', 'Quarter']) 
    top_NB_df = top_NB_df.reset_index()
    for y in top_NB_df.index:
        if y == 0:
            Q1_15 = top_NB_df['Sales Number'].loc[y]
        if y == 1:
            Q2_15 = top_NB_df['Sales Number'].loc[y]
        if y == 2:
            Q3_15 = top_NB_df['Sales Number'].loc[y]
        if y == 3:
            Q4_15 = top_NB_df['Sales Number'].loc[y]
        if y == 4:
            Q1_16 = top_NB_df['Sales Number'].loc[y]
        if y == 5:
            Q2_16 = top_NB_df['Sales Number'].loc[y]
        if y == 6:
            Q3_16 = top_NB_df['Sales Number'].loc[y]
        if y == 7:
            Q4_16 = top_NB_df['Sales Number'].loc[y]
        if y == 8:
            Q1_17 = top_NB_df['Sales Number'].loc[y]
        if y == 9:
            Q2_17 = top_NB_df['Sales Number'].loc[y]
        if y == 10:
            Q3_17 = top_NB_df['Sales Number'].loc[y]
        if y == 11:
            Q4_17 = top_NB_df['Sales Number'].loc[y]
        if y == 12:
            Q1_18 = top_NB_df['Sales Number'].loc[y]
        if y == 13:
            Q2_18 = top_NB_df['Sales Number'].loc[y]
        if y == 14:
            Q3_18 = top_NB_df['Sales Number'].loc[y]
        if y == 15:
            Q4_18 = top_NB_df['Sales Number'].loc[y]
    top_NB_arr.append((name, Q1_15, Q2_15, Q3_15, Q4_15, Q1_16, Q2_16, Q3_16, Q4_16, 
                    Q1_17, Q2_17, Q3_17, Q4_17, Q1_18, Q2_18, Q3_18, Q4_18))

## Get the Mean Sales Price per quarter and year of top LGA for graphing
top_nbmean_arr=[]

for x in range(len(top_NB)):
    name = top_NB[x]
    top_nbmean_df = sales_df[sales_df['LGA']==top_NB[x]].sort_values(by="Mean Sales Price", ascending = False)
    top_nbmean_df = top_nbmean_df[top_nbmean_df['Dwelling Type'] == 'Total']
    top_nbmean_df = top_nbmean_df.sort_values(by=['Year', 'Quarter']) 
    top_nbmean_df = top_nbmean_df.reset_index()
    for y in top_nbmean_df.index:
        if y == 0:
            Q1_15 = top_nbmean_df['Mean Sales Price'].loc[y]
        if y == 1:
            Q2_15 = top_nbmean_df['Mean Sales Price'].loc[y]
        if y == 2:
            Q3_15 = top_nbmean_df['Mean Sales Price'].loc[y]
        if y == 3:
            Q4_15 = top_nbmean_df['Mean Sales Price'].loc[y]
        if y == 4:
            Q1_16 = top_nbmean_df['Mean Sales Price'].loc[y]
        if y == 5:
            Q2_16 = top_nbmean_df['Mean Sales Price'].loc[y]
        if y == 6:
            Q3_16 = top_nbmean_df['Mean Sales Price'].loc[y]
        if y == 7:
            Q4_16 = top_nbmean_df['Mean Sales Price'].loc[y]
        if y == 8:
            Q1_17 = top_nbmean_df['Mean Sales Price'].loc[y]
        if y == 9:
            Q2_17 = top_nbmean_df['Mean Sales Price'].loc[y]
        if y == 10:
            Q3_17 = top_nbmean_df['Mean Sales Price'].loc[y]
        if y == 11:
            Q4_17 = top_nbmean_df['Mean Sales Price'].loc[y]
        if y == 12:
            Q1_18 = top_nbmean_df['Mean Sales Price'].loc[y]
        if y == 13:
            Q2_18 = top_nbmean_df['Mean Sales Price'].loc[y]
        if y == 14:
            Q3_18 = top_nbmean_df['Mean Sales Price'].loc[y]
        if y == 15:
            Q4_18 = top_nbmean_df['Mean Sales Price'].loc[y]
    top_nbmean_arr.append((name, Q1_15, Q2_15, Q3_15, Q4_15, Q1_16, Q2_16, Q3_16, Q4_16, 
                    Q1_17, Q2_17, Q3_17, Q4_17, Q1_18, Q2_18, Q3_18, Q4_18))

## Graph LGA
top_NB_df = pd.DataFrame.from_records(top_NB_arr)
top_NB_df.columns = ['LGA', '2015.Q1', '2015.Q2', '2015.Q3', '2015.Q4', '2016.Q1', '2016.Q2', '2016.Q3', '2016.Q4',
                      '2017.Q1', '2017.Q2', '2017.Q3', '2017.Q4', '2018.Q1', '2018.Q2', '2018.Q3', '2018.Q4']
top_NB_df.index = top_NB_df['LGA']
top_NB_df = top_NB_df.drop('LGA', axis=1)
## remove columns for the graph to look nice
top_NB_df = top_NB_df.drop(['2015.Q1', '2015.Q2', '2015.Q3', '2015.Q4', '2016.Q1', '2016.Q2', '2016.Q3', '2016.Q4',
                            '2017.Q1'], axis=1)

## MEAN 
## Graph LGA
top_nbmean_df = pd.DataFrame.from_records(top_nbmean_arr)
top_nbmean_df.columns = ['LGA', '2015.Q1', '2015.Q2', '2015.Q3', '2015.Q4', '2016.Q1', '2016.Q2', '2016.Q3', '2016.Q4',
                      '2017.Q1', '2017.Q2', '2017.Q3', '2017.Q4', '2018.Q1', '2018.Q2', '2018.Q3', '2018.Q4']
top_nbmean_df.index = top_nbmean_df['LGA']
top_nbmean_df = top_nbmean_df.drop('LGA', axis=1)

### Dwelling Type Most People Buy ###
## dataframe per dwelling type
S_df = sales_df[sales_df['Dwelling Type']=='Strata']
NS_df = sales_df[sales_df['Dwelling Type']=='Non Strata']

## get total count per year
year = [2015, 2016, 2017, 2018]
bond_count_arr = []
for x in range(len(year)):
    yr = year[x]
    strata = S_df[S_df['Year']==yr]['Sales Number'].sum()
    name = 'Strata'
    bond_count_arr.append((yr, strata, name))
    ns = NS_df[NS_df['Year']==yr]['Sales Number'].sum()
    name = 'Non Strata'
    bond_count_arr.append((yr, ns, name))

### DF for dwelling type sales count ###
bond_count_df = pd.DataFrame.from_records(bond_count_arr)
bond_count_df.columns = ['Year', 'Count', 'Dwelling Type']