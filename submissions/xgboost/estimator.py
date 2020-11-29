import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.impute import SimpleImputer
import xgboost
import datetime as dt
import holidays
from geopy.distance import geodesic

def _nextworkday(date):
    one_day = dt.timedelta(days=1)
    next_day = date + one_day 
    while next_day.weekday() in holidays.WEEKEND or next_day in Holidays_US:
        next_day += one_day 
    return (next_day - date).days

def _lastworkday(date):
    one_day = dt.timedelta(days=1)
    last_day = date - one_day 
    while last_day.weekday() in holidays.WEEKEND or last_day in Holidays_US:
        last_day -= one_day 
    return (date - last_day).days   


def _merge_external_data(X):
    import datetime as dt
    import holidays

    #get holidays 
    Holidays_US = holidays.US()[dt.date(2011,7, 1):dt.date(2013,6, 5)] + holidays.US()[dt.date(2012,1, 1):dt.date(2012,12, 31)]

    filepath = os.path.join(
            os.path.dirname(__file__), 'external_data.csv')

    external = pd.read_csv(filepath)
    external.loc[:,"Date"] = pd.to_datetime(external.loc[:,"Date"])

    #deal with data format
    external['Precipitationmm'].replace('T',0.0, inplace=True)
    external['Precipitationmm'] = external['Precipitationmm'].astype('float')
    for i in range(len(external['POP'])):
        external['POP'][i] = external.loc[:,'POP'][i].replace(',','')
    external['POP'] = external['POP'].astype('int')   
    external.drop(columns=['Events', 'City', 'StateCodes'],inplace=True)

    external.drop(columns =['Year','Region'],inplace=True)

    # define the departure and arrival dataframe
    col_dep = ['d_' + name for name in list(external.columns)]
    col_arr = [w.replace('d_', 'a_') for w in col_dep]

    # adjust the column name for merge
    col_dep = [w.replace('d_AirPort', 'Departure') for w in col_dep]
    col_dep = [w.replace('d_Date', 'DateOfDeparture') for w in col_dep]
    col_arr = [w.replace('a_AirPort', 'Arrival') for w in col_arr]
    col_arr = [w.replace('a_Date', 'DateOfDeparture') for w in col_arr]

    # 
    d_external = external.copy()
    a_external = external.copy()

    # rename the column
    d_external.columns = col_dep
    a_external.columns = col_arr

     # merge with X_encoded
    X_encoded = X.copy()
    X_encoded.loc[:,'DateOfDeparture'] = pd.to_datetime(X_encoded.loc[:,'DateOfDeparture'])
    X_encoded = pd.merge(X_encoded, d_external, how='left', on=['DateOfDeparture', 'Departure'],
                        sort=False)
    X_encoded = pd.merge(X_encoded, a_external, how='left', on=['DateOfDeparture', 'Arrival'],
                        sort=False)
    #

    X_encoded['nextworkday']=0
    X_encoded['lastworkday']=0


    for i in range(len(X_encoded)):
        X_encoded['nextworkday'][i] = _nextworkday(X_encoded.loc[:,'DateOfDeparture'][i])
        X_encoded['lastworkday'][i] = _lastworkday(X_encoded.loc[:,'DateOfDeparture'][i])
        

    # compute geographic distance
    X_encoded["Distance"] = X_encoded.apply(
            lambda x: geodesic((x["d_Latitude"],x["d_longitude"]),(x["a_Latitude"],x["a_longitude"])).km, axis=1)

    # split year, month and etc.
    X_encoded['year'] = X_encoded.loc[:,'DateOfDeparture'].dt.year
    X_encoded['month'] = X_encoded.loc[:,'DateOfDeparture'].dt.month
    X_encoded['day'] = X_encoded.loc[:,'DateOfDeparture'].dt.day
    X_encoded['weekday'] = X_encoded.loc[:,'DateOfDeparture'].dt.weekday
    X_encoded['week'] = X_encoded.loc[:,'DateOfDeparture'].dt.week
    X_encoded['n_days'] = X_encoded.loc[:,'DateOfDeparture'].apply(lambda date: 
                                                                     (date - pd.to_datetime("1970-01-01")).days)

    X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
    X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
    X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
    X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
    X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))

    # drop the original data
    X_encoded.drop(columns=['Departure','Arrival','DateOfDeparture',
                            'd_Unnamed: 0','d_Unnamed: 0.1','a_Unnamed: 0','a_Unnamed: 0.1',
                           'd_coordinates','a_coordinates','d_State','a_State',
                           'd_iso_region','a_iso_region'], inplace=True)


    for column in list(X_encoded.columns[X_encoded.isnull().sum() > 0]):
        mean_val = X_encoded[column].mean()
        X_encoded[column].fillna(mean_val, inplace=True)
      
    X_selected = X_encoded[['WeeksToDeparture', 'd_Min TemperatureC', 'd_Total_ops',
       'd_Total Delays', 'd_Actual Departures', 'd_Actual Arrivals',
       'd_Delayed Arrivals', 'd_Price', 'd_elevation_ft', 'd_Division',
       'd_StPOP', 'd_StRBirth', 'd_StRDeath', 'd_StRMig', 'd_GDP', 'd_POP',
       'd_RPI', 'd_Latitude', 'd_longitude', 'a_Total_ops', 'a_Total Delays',
       'a_Actual Departures', 'a_Actual Arrivals', 'a_Delayed Arrivals',
       'a_elevation_ft', 'a_Division', 'a_StPOP', 'a_StRBirth', 'a_StRDeath',
       'a_StRMig', 'a_GDP', 'a_POP', 'a_RPI', 'a_Latitude', 'a_longitude',
       'nextworkday', 'lastworkday','Distance', 'month', 'day', 'weekday',
       'week', 'n_days', 'm_10', 'wd_1', 'wd_3', 'wd_5', 'wd_6', 'w_1', 'w_14',
       'w_20', 'w_21', 'w_27', 'w_35', 'w_47', 'w_52']]
    
    return X_selected

def get_estimator():
    date_merger = FunctionTransformer(_merge_external_data)
    regressor = xgboost.XGBRegressor(colsample_bytree=0.7,
                     gamma=0.3,                 
                     learning_rate=0.07,
                     max_depth=5,
                     min_child_weight=3,
                     n_estimators=2000,                                                                    
                     reg_alpha=0.75,
                     reg_lambda=0.5,
                     subsample=0.6,
                     seed=42) 

    return make_pipeline(date_merger, regressor)
