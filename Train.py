import numpy as np
import pandas as pd

from DataProcessing import process
from Util import log_GS_res

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

# Train Naive Bayes method
def train_naive_bayes(X, Y, cv):
    # Define parameters for grid search
    parameters = [{'var_smoothing': np.logspace(-7, -11, num=100)}]

    # Use grid search to determine best model
    model = GaussianNB()
    grid_search = GridSearchCV(model, parameters, cv=cv, scoring='accuracy')
    grid_search.fit(X, Y)

    return grid_search.best_params_, grid_search.best_score_, grid_search.best_estimator_

# Train Logistic Regressor
def train_logistic_regressor(X, Y, cv):
    # Define parameters for grid search
    parameters = [{'C': np.logspace(-3, 2, num=6),
                   'penalty': ['none', 'l1', 'l2', 'elasticnet'],
                   'class_weight': [None, 'balanced'],
                   'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}]

    # Use grid search to determine best model
    model = LogisticRegression()
    grid_search = GridSearchCV(model, parameters, cv=cv, scoring='accuracy')
    grid_search.fit(X, Y)

    return grid_search.best_params_, grid_search.best_score_, grid_search.best_estimator_

# Train random forest
def train_random_forest(X, Y, cv):
    # Define parameters for grid search
    parameters = [{'max_depth': [None, 5, 10, 15],
                   'min_samples_split': [2, 3, 4, 5],
                   'n_estimators': [10, 50, 100, 500]}]

    # Use grid search to determine best model
    model = RandomForestClassifier()
    grid_search = GridSearchCV(model, parameters, cv=cv, scoring='accuracy')
    grid_search.fit(X, Y)

    return grid_search.best_params_, grid_search.best_score_, grid_search.best_estimator_

if __name__ == "__main__":
    # Import data from CSV
    training_data = pd.read_csv('Data/train.csv')
    test_data = pd.read_csv('Data/test.csv')

    # Process data
    train_data, test_data = process(training_data, test_data, val=False)
    X, Y = train_data

    # Train sklearn models
    cross_validation_folds = 5

    NBparams, NBscore, _ = train_naive_bayes(X, Y, cross_validation_folds)
    LRparams, LRscore, _ = train_logistic_regressor(X, Y, cross_validation_folds)
    RFparams, RFscore, _ = train_random_forest(X, Y, cross_validation_folds)

    log_GS_res(NBparams, NBscore, LRparams, LRscore, RFparams, RFscore)

    print("Done")