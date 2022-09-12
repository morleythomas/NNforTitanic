import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from Util import Imputer, CabinCounter, CabinLetter, OneHotter, Dropper
from Util import create_val_split, create_X_Y_split, scale_test_set

def process(train, test, val=True):

    # Create validation set
    if val:
        train, val = create_val_split(train, ['Survived', 'Pclass', 'Sex'])

    # Create new columns in dataframes
    counter = CabinCounter()
    letter = CabinLetter()

    train = counter.fit_transform(X=train)
    test = counter.fit_transform(X=test)

    train = letter.fit_transform(X=train)
    test = letter.fit_transform(X=test)

    if val:
        val = counter.fit_transform(X=val)
        val = letter.fit_transform(X=val)

    # Instantiate pipeline
    data_pipeline = Pipeline([("Imputer", Imputer()),
                              ("OneHotter", OneHotter()),
                              ("Dropper", Dropper())])

    # Transform data
    train = data_pipeline.fit_transform(train)
    test = data_pipeline.fit_transform(test)
    if val:
        val = data_pipeline.fit_transform(val)

    # Split train and val into data and targets
    X_train, Y_train = create_X_Y_split(train, 'Survived')
    if val:
        X_val, Y_val = create_X_Y_split(val, 'Survived')

    # Scale test set
    test = scale_test_set(test)

    if val:
        return [X_train, Y_train], [X_val, Y_val], test

    return [X_train, Y_train], test

