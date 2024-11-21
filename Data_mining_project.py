import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def drop_Columns(df):# drop irrelevant columns
    df.drop(columns=['Doctor','Hospital','Name','Room Number','Discharge Date','Date of Admission'],inplace=True)
    return df
def check_date(df): # cross field validation
    df.loc[df['Discharge Date'] < df['Date of Admission'], 'Discharge Date'] = pd.NA
    print('There are ',df['Discharge Date'].isnull().sum(),'missing date')
    return df

def check_consistency(df_healthCare): # consistency 
    print(df_healthCare['Blood Type'].str.strip().value_counts().to_string())  
    print(df_healthCare['Admission Type'].str.strip().value_counts().to_string())
    print(df_healthCare['Medication'].str.strip().value_counts().to_string())
    print(df_healthCare['Insurance Provider'].str.strip().value_counts().to_string())
    print(df_healthCare['Medical Condition'].str.strip().value_counts().to_string())
    print(df_healthCare['Gender'].str.strip().value_counts().to_string())
    print(df_healthCare['Age'].min(),df_healthCare['Age'].max()) # check range age 
    return df_healthCare

def check_correlation(df_correlation): # correlation between amount of days[end -start] ,billing amount, and Room number 
    df_healthCare['diff_date']=df_healthCare['Discharge Date']-df_healthCare['Date of Admission']
    df=df_healthCare[['diff_date','Billing Amount','Room Number']].corr()
    print(df)
    return df_correlation

def make_null(df):
    percentage_null = 0.2
    num_nulls = int(len(df) * percentage_null)
    indices = np.random.choice(df.index, size=num_nulls, replace=False)
    df.loc[indices, 'Age'] = np.nan
    return df

def handling_missing_values(df):
    #inditialize dictionary for medication
    medication_dict = {}
    for medication_name,frame_name in df.groupby('Medication'):
    
        Q1=df['Age'].quantile(.25)
        Q3=df['Age'].quantile(.75)
        IQR=Q3-Q1
        lower_bound=Q1-1.5*IQR
        upper_bound=Q3+1.5*IQR
        filtered_age=df.loc[(df['Age'] <=upper_bound)&((df['Age'] >=lower_bound)),'Age']
        # print(frame_name.head(10).to_string(),'\n','========',filtered_age)
        medication_dict[medication_name]=filtered_age.mean()

    # for x,y in medication_dict.items(): for printing if you need
        # print(x,y)
    df['Age']=df['Age'].fillna(df['Medication'].map(medication_dict))  
    return df



def scalling (df):
    scaler=StandardScaler(copy=True,with_mean=True,with_std=True) #just you say, and #initialize an object
    scalling=['Age','Billing Amount']
    scaler.fit(df[scalling]) # apply what I said such as mean and standard deviation
    res=scaler.transform(df[scalling])# calc (x-M)/var
    # print(res)
    df[scalling]=res
    return df

df_healthCare=pd.read_csv('H:\dataSet\healthcare_dataset.csv')
df_healthCare=make_null(df_healthCare)
df_healthCare.to_csv('H:\\dataSet\\missingValue_Health_Care.csv',index=False)    

print(df_healthCare.isnull().sum())
# print(df_healthCare.info())
t=df_healthCare.columns
# print(t)
df_healthCare['Discharge Date']=pd.to_datetime(df_healthCare['Discharge Date'],errors='coerce')
df_healthCare['Date of Admission']=pd.to_datetime(df_healthCare['Date of Admission'],errors='coerce')

check_date(df_healthCare)
df_healthCare=check_consistency(df_healthCare)          #check consistency
df_healthCare=check_correlation(df_healthCare)          #check correlation
df_healthCare=drop_Columns(df_healthCare)               # drop the columns

print(df_healthCare.isnull().sum())


print(df_healthCare.isnull().sum()/len(df_healthCare)*100) # print info including 20% percentage of nulls
df_healthCare=handling_missing_values(df_healthCare)

df_healthCare=scalling(df_healthCare)                       # scalling
print(df_healthCare)                                        #print data frame
print(df_healthCare.isnull().sum()/len(df_healthCare)*100) # print info including 0 % percentage of nulls
