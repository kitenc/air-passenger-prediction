import numpy as np
import pandas as pd


class feature_merge():
    def __init__(self):
        pass
        
    def fit_transform(self, X):
        external = pd.read_csv(r"submissions\use_external_data\external_data_mod.csv", header=0)
        external.loc[:,"Date"] = pd.to_datetime(external.loc[:,"Date"])
        
        # define the departure and arrival dataframe
        col_dep = ['d_' + name for name in list(external_data.columns)]
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
        
        # compute geographic distance
        X_encoded["Distance"] = X_encoded.apply(
                lambda x: geodesic(
                (x["d_latitude"],x["d_longitude"]),(x["a_latitude"],x["a_longitude"])
                ).km, axis=1)

        X_encoded = X_encoded.join(pd.get_dummies(X_encoded.loc[:,'Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded.loc[:,'Arrival'], prefix='a'))
        
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
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
        X_encoded = X_encoded.drop('DateOfDeparture',axis = 1)
        return X_encoded