# Author: Ryan Varnell
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from ModelUtility import fit_and_test


# Random Forest Regression ML model.
def main():
    # Load data.
    data = pd.read_csv('data/wade.csv', index_col='timestamp', parse_dates=True)

    # Separate our chosen independent variables and create a new dataframe with only our desired data.
    in_var = ['DO_mgL', 'NO3_mgNL', 'DO_sat', 'spCond', 'dewPoint']
    selected_data = data[['temp'] + in_var]
    X = selected_data[in_var]
    y = selected_data['temp']

    # Fit RF regression model to data set.
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    fit_and_test(model, X, y)


if __name__ == '__main__':
    main()
