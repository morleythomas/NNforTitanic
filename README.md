# NNforTitanic

## Abstract
This project willl explore the efficacy of a neural network approach to Kaggle's Titanic dataset competition by comparing evaluation metrics with other machine learning techniques.

## Problem
The rows in our dataset represent indiviuals who embarked on the Titanic's maiden voyage in 1912. The columns represent particular details about each individual (ie, name, age, ticket class). The dependant variable we wish to predict is whether or not a given passenger survived the voyage, or died during the disaster. Hence, this is a binary classification problem.

## Method 
To establish a baseline, Sci-kit Learn models `RandomForestClassifier`, `LogisticRegression` and `GaussianNB` are used. For a baseline, the highest accuracy score acquired with `GridSearchCV` is used. We then rank these scores into a table; and, once trained, we can acquire a validation score for our neural netowrk approach, and subsequently see where it lands on the scoreboard. 

The neural network is a feedforward network consisting of one hidden layer of 50 hidden units and a ReLU activation, followed a by an output layer of one sigmoid unit. Predictions are taken on a threshold basis, ie for outputs lower than 0.5, we predict 0 (died); and for ouputs higher we predict 1 (survived).

## The data

## Results
