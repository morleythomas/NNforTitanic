import torch

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

'''
Pipeline classes
'''
#Impute age feature using mean
class Imputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['Age'] = SimpleImputer(strategy='mean').fit_transform(X[['Age']])
        return X

#Create column for number of cabins held
class CabinCounter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['CabinCount'] = X.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
        return X

#Create column for cabin first letter
class CabinLetter(BaseEstimator, TransformerMixin):

    def fit (self, X, y=None):
        return self

    def transform(self, X):
        X['CabinLetter'] = X.Cabin.apply(lambda x: str(x)[0])
        return X

#Create one hot encodings class
class OneHotter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # Instantiate encoders
        embarked_columns = ['C', 'Q', 'S', 'N']
        sex_columns = ['female', 'male']
        cletter_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'n']
        self.one_hot_embarked = OneHotEncoder()
        self.one_hot_sex = OneHotEncoder()
        self.one_hot_cletter = OneHotEncoder(categories=[cletter_columns])

        # Encode embarked feature
        embarked = self.one_hot_embarked.fit_transform(X[['Embarked']]).toarray()

        for i in range(len(embarked.T)):
            X[embarked_columns[i]] = embarked.T[i]

        # Encode sex feature
        sex = self.one_hot_sex.fit_transform(X[['Sex']]).toarray()

        for i in range(len(sex.T)):
            X[sex_columns[i]] = sex.T[i]

        # Encode cabin letters
        cletter = self.one_hot_cletter.fit_transform(X[['CabinLetter']]).toarray()

        for i in range(len(cletter.T)):
            X[cletter_columns[i]] = cletter.T[i]

        return X

#Drop columns from df that are not needed
class Dropper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(['Embarked', 'Sex', "Cabin", "CabinLetter", 'N', "PassengerId", "Name", "Ticket"], axis=1, errors="ignore")

'''
Functions
'''
# Create validation dataset with even splits across given collumns
def create_val_split(dataset, even_cols):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=6)

    for train, val in splitter.split(dataset, dataset[even_cols]):
        train_set = dataset.loc[train]
        val_set = dataset.loc[val]

    return train_set, val_set

# Split dataset into data and labels, then scale data
def create_X_Y_split(dataset, Y_col):
    scaler = StandardScaler()
    X = dataset.drop([Y_col], axis=1)
    Y = dataset[Y_col]

    X = scaler.fit_transform(X)
    Y = Y.to_numpy(Y)

    return X, Y

# Scale test set
def scale_test_set(dataset):
    scaler = StandardScaler()
    return scaler.fit_transform(dataset)

# Log grid search results
def log_results(NBparams, NBscore, LRparams, LRscore, RFparams, RFscore, NNloss, NNreport):
    with open('log.txt', 'w') as f:

        f.write(" ----- Naive Bayes -----\n\n")
        f.write("Best params:\n")
        for param in NBparams:
            f.write("{}: {}\n".format(param, NBparams[param]))
        f.write("\nScore: {}\n\n\n".format(NBscore))

        f.write(" ----- Logistic Regression -----\n\n")
        f.write("Best params:\n")
        for param in LRparams:
            f.write("{}: {}\n".format(param, LRparams[param]))
        f.write("\nScore: {}\n\n\n".format(LRscore))

        f.write(" ----- Random Forest -----\n\n")
        f.write("Best params:\n")
        for param in RFparams:
            f.write("{}: {}\n".format(param, RFparams[param]))
        f.write("\nScore: {}\n\n\n".format(RFscore))

        f.write(" ----- Neural Network ------\n\n")
        f.write("Lowest validation loss: {}\n".format(NNloss))
        print("\n", NNreport, file=f)
        f.write("\n\n\n")

        f.close()

# Produce chart showing val loss and train loss across epochs for NN approach
def get_chart(train_list, val_list):
    plt.plot(train_list)
    plt.plot(val_list)
    plt.legend(("Training loss", "Validation loss"))
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross Entropy loss")
    plt.show()

# Evaluate for accuracy on an evaluation set
def evaluate(model, dataset):
    X, Y = dataset

    # Tensorise data
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    # Get predictions
    preds = model(X.reshape(1, X.shape[0], X.shape[1]).float()).flatten()
    preds = list(map(lambda x: 0 if x < 0.5 else 1, preds.tolist()))

    return (classification_report(Y.tolist(), preds, labels=[0, 1], target_names=["Died", "Survived"]))