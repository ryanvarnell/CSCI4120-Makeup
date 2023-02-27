import re
from math import sqrt, floor
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedKFold

# Regex for getting a model name from the model object.
# Putting it here, so we don't have to compile every time we need it.
regex = re.compile("[^a-zA-Z]")


# Divides X and y into training and testing sets, fits the model to the sets, and compares the predicted values
# of the model to the testing set with a graph visual.
def fit_and_test(model, X, y):
    model_name = name(model)
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)

    # Divide the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fit the model to the training sets.
    model.fit(X_train, y_train)

    # Get predictions and plot to compare with test set.
    y_prediction = model.predict(X_test)
    compare_plot(model, y_test, y_prediction)

    cv_score = cross_val_score(model, X, y, cv=cv)
    rmse_score = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)

    if model_name == 'Lasso':
        print(model_name + ' alpha: ' + str(model.get_params()['alpha']))
    print(model_name + " CV scores: " + str(cv_score))
    print(model_name + " mean CV score: " + str(np.mean(cv_score)))
    print(model_name + " RMSE score: " + str(rmse_score))
    print(model_name + " mean RMSE score: " + str(np.mean(rmse_score)))


# Get the name of a model. Probably unnecessary for our use-case here, but it was fun to do.
# It also doesn't work for the Ridge model, as it sees it as Lasso.
def name(model):
    model_name = ''
    for token in re.findall("[A-Z][^A-Z]*", str(type(model)).split('.')[-1]):
        model_name = model_name + ' ' + regex.sub('', token)
    return model_name.strip()


# Takes the predictive y-values of a model and compares them to the test y-values with a graph visual.
def compare_plot(model, y_test, y_prediction):
    # We're grabbing only a portion of the data for visualization, as it's hard to see the data otherwise.
    # The size of the portion is the floor of the square root of the total size of the set.
    plt.plot(range(floor(sqrt(len(y_prediction)))), y_test[:floor(sqrt(len(y_test)))], alpha=0.5,
             label="Reference Data")
    plt.plot(range(floor(sqrt(len(y_prediction)))), y_prediction[:floor(sqrt(len(y_prediction)))], alpha=0.5,
             label="Predicted Data")
    plt.title('Predicted Temperature')
    plt.ylabel('Temperature')
    plt.legend()

    # Add appropriate suptitle depending on model type.
    plt.suptitle(name(model))

    plt.show()


# Tune models that require it.
def tune(model, X, y):
    parameters = {'alpha': arange(0.01, 1, 0.01)}
    cv = GridSearchCV(model, parameters, cv=5, n_jobs=-1)
    search = cv.fit(X, y)
    alpha = search.best_estimator_.alpha
    return Lasso(alpha=alpha)
