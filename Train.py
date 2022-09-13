import random
import torch

import numpy as np
import pandas as pd

from DataProcessing import process
from Network import Network
from Util import log_results, get_chart, evaluate

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from torch.utils.data import DataLoader, TensorDataset

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
                   'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                   "random_state": [5]}]

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
                   'n_estimators': [10, 50, 100, 500],
                   "random_state": [5]}]

    # Use grid search to determine best model
    model = RandomForestClassifier()
    grid_search = GridSearchCV(model, parameters, cv=cv, scoring='accuracy')
    grid_search.fit(X, Y)

    return grid_search.best_params_, grid_search.best_score_, grid_search.best_estimator_

# Seed worker for data loader random seed
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

# Train nueral network
def train_nn(train, val):

    # Define hparams
    learning_rate = 0.00005
    weight_decay = 0.002
    batch_size = 32
    n_epochs = 2000

    # Tensorise data
    X_train, Y_train = train
    X_val, Y_val = val

    X_train = torch.tensor(X_train)
    Y_train = torch.tensor(Y_train)
    X_val = torch.tensor(X_val)
    Y_val = torch.tensor(Y_val)

    train = TensorDataset(X_train, Y_train)

    # Init network & training components
    torch.manual_seed(0)
    model = Network(X_train.shape[1])
    criterion = torch.nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loss_list = []
    val_loss_list = []
    prev_loss = 1

    # Random seed and data loader initialisation
    g = torch.Generator()
    g.manual_seed(0)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(n_epochs):

        if epoch % 100 == 0:
            print("Epoch: ", epoch)

        for idx, batch in enumerate(loader):
            optimiser.zero_grad()
            preds = model(batch[0].float()).flatten()
            J = criterion(preds.float(), batch[1].float())
            J.backward()
            optimiser.step()

        with torch.no_grad():
            # Get loss data from total train set
            train_res = model(X_train.reshape(1, X_train.shape[0], X_train.shape[1]).float()).flatten()
            train_loss = criterion(train_res.float(), Y_train.float())
            train_loss_list.append(train_loss.item())

            # Get loss data from total val set
            val_res = model(X_val.reshape(1, X_val.shape[0], X_val.shape[1]).float()).flatten()
            val_loss = criterion(val_res.float(), Y_val.float())
            val_loss_list.append(val_loss.item())

            # Save best model
            if val_loss.item() < prev_loss:
                prev_loss = val_loss.item()
                best_model = model

    # Create chart of training process
    get_chart(train_loss_list, val_loss_list)

    return best_model, prev_loss

if __name__ == "__main__":

    # Import data from CSV
    training_data = pd.read_csv('Data/train.csv')
    test_data = pd.read_csv('Data/test.csv')

    # Process data
    train_data, _ = process(training_data, test_data, val_set=False)
    X, Y = train_data

    # Train sklearn models
    cross_validation_folds = 5

    NBparams, NBscore, _ = train_naive_bayes(X, Y, cross_validation_folds)
    LRparams, LRscore, _ = train_logistic_regressor(X, Y, cross_validation_folds)
    RFparams, RFscore, _ = train_random_forest(X, Y, cross_validation_folds)

    # Process data & train neural network
    train_data, val_data, _ = process(training_data, test_data)
    model, NNloss = train_nn(train_data, val_data)
    NNreport = evaluate(model, val_data)

    # Log final results data
    log_results(NBparams, NBscore, LRparams, LRscore, RFparams, RFscore, NNloss, NNreport)