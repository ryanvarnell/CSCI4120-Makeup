# Author: Ryan Varnell
import re
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from ModelUtility import fit_and_test, tune

# Regex for getting a model name from the model object.
# Putting it here, so we don't have to compile every time we need it.
regex = re.compile("[^a-zA-Z]")


def main():
    # Load the provided data
    data = pd.read_csv('data/wade.csv', index_col='timestamp', parse_dates=True)

    # This big block comment is for all of the data analysis included in the report.
    #
    # data.info()
    # Output:
    # /---------------------------------------------------------------------\
    # DatetimeIndex: 1660 entries, 2017-10-21 13:45:00 to 2017-11-10 06:30:00
    # Data columns (total 21 columns):
    #  #   Column           Non-Null Count  Dtype
    # ---  ------           --------------  -----
    #  0   q_cms            1660 non-null   float64
    #  1   NO3_mgNL         1660 non-null   float64
    #  2   SRP_mgPL         1660 non-null   float64
    #  3   DO_mgL           1660 non-null   float64
    #  4   DO_sat           1660 non-null   float64
    #  5   fDOM             1660 non-null   float64
    #  6   pH               1660 non-null   float64
    #  7   spCond           1660 non-null   float64
    #  8   temp             1660 non-null   float64
    #  9   turb             1660 non-null   float64
    #  10  precip_mm        1660 non-null   float64
    #  11  temp_C           1660 non-null   float64
    #  12  atm_mbar         1660 non-null   float64
    #  13  PAR_uE           1660 non-null   int64
    #  14  windSpeed        1660 non-null   float64
    #  15  gustSpeed        1660 non-null   float64
    #  16  solarRad_wm2     1660 non-null   int64
    #  17  windDir          1660 non-null   int64
    #  18  Rh               1660 non-null   float64
    #  19  dewPoint         1660 non-null   float64
    #  20  Basic_Threshold  1660 non-null   int64
    # dtypes: float64(17), int64(4)
    # memory usage: 285.3 KB
    # \---------------------------------------------------------------------/
    #
    # Calculate the interquartile range for use in identifying extreme outliers.
    # This will be useful for cleaning up the data if necessary.
    # spread = data.describe().T
    # IQR = spread['75%'] - spread['25%']
    #
    # Grab the outliers and stick them in a new 'outliers' column.
    # An extreme outlier is any value that is three IRs above or below the 75th or 25th percentile, respectively.
    # spread['outliers'] = (spread['min'] < (spread['25%'] - (3 * IQR))) | (spread['max'] > (spread['75%'] + 3 * IQR))
    #
    # Check which features contain extreme outliers
    # print(spread.loc[spread.outliers, ])
    # Output:
    # /--------------------------------------------------------------------------------------------------------\
    #                count       mean         std       min      25%       50%        75%          max  outliers
    # SRP_mgPL      1660.0   0.002986    0.002071  0.000160  0.00152  0.002417   0.004041     0.012995      True
    # turb          1660.0   1.126657    3.389062  0.080667  0.24650  0.363333   0.752667    62.535333      True
    # precip_mm     1660.0   0.089277    0.338782  0.000000  0.00000  0.000000   0.000000     6.800000      True
    # PAR_uE        1660.0  65.742169  140.811254  1.000000  1.00000  1.000000  69.000000  1024.000000      True
    # windSpeed     1660.0   0.313855    0.482035  0.000000  0.00000  0.000000   0.500000     2.500000      True
    # solarRad_wm2  1660.0  32.778313   71.155572  1.000000  1.00000  1.000000  32.000000   539.000000      True
    # \--------------------------------------------------------------------------------------------------------/
    #
    # We can get a visual for the above features individually with:
    # plt.rcParams['figure.figsize'] = [14, 8]
    # data.featureName.hist()
    # plt.show()
    #
    # Look for correlations between 'precip_mm' and the other features using the Pearson correlation coefficient.
    # Drop anything below absolute value 0.6, with leniency.
    # print(data.corr()[['temp']].sort_values('temp'))
    # Output:
    # /-----------------------\
    #                      temp
    # DO_mgL          -0.972560
    # NO3_mgNL        -0.897186
    # DO_sat          -0.778274
    # atm_mbar        -0.435633
    # Basic_Threshold -0.217459
    # solarRad_wm2    -0.077421
    # windDir         -0.073441
    # PAR_uE          -0.073142
    # fDOM            -0.011512
    # q_cms            0.068921
    # gustSpeed        0.074531
    # windSpeed        0.106290
    # precip_mm        0.148349
    # turb             0.149258
    # Rh               0.184314
    # SRP_mgPL         0.480550
    # pH               0.487443
    # spCond           0.584501
    # temp_C           0.807261
    # dewPoint         0.879371
    # temp             1.000000
    # \-----------------------/

    # Separate our chosen independent variables and create a new dataframe with only our desired data.
    in_var = ['DO_mgL', 'NO3_mgNL', 'DO_sat', 'spCond', 'dewPoint']
    selected_data = data[['temp'] + in_var]
    X = selected_data[in_var]
    y = selected_data['temp']

    # Build basic Linear Regression model.
    lr_model = LinearRegression(fit_intercept=False)
    fit_and_test(lr_model, X, y)

    # Build and tune Lasso model.
    lasso_model = Lasso(fit_intercept=False, max_iter=5000)
    lasso_model = tune(lasso_model, X, y)
    fit_and_test(lasso_model, X, y)

    # Build and tune Ridge model.
    print('***************** Everything below here is Ridge *****************')
    ridge_model = Ridge(fit_intercept=False)
    ridge_model = tune(ridge_model, X, y)
    fit_and_test(ridge_model, X, y)


if __name__ == '__main__':
    main()
