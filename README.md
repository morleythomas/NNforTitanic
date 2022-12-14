# NNforTitanic

## Abstract
This project willl explore the efficacy of a neural network approach to Kaggle's Titanic dataset competition by comparing evaluation metrics with other machine learning techniques.

## Problem
The rows in our dataset represent indiviuals who embarked on the Titanic's maiden voyage in 1912. The columns represent particular details about each individual (ie, name, age, ticket class). The dependant variable we wish to predict is whether or not a given passenger survived the voyage, or died during the disaster. Hence, this is a binary classification problem.

## Method 
To establish a baseline, Sci-kit Learn models `RandomForestClassifier`, `LogisticRegression` and `GaussianNB` are used. For a baseline, the highest accuracy score acquired with `GridSearchCV` is used. We then rank these scores into a table; and, once trained, we can acquire a validation score for our neural netowrk approach, and subsequently see where it lands on the scoreboard. 

The neural network is a feedforward network consisting of one hidden layer of 50 hidden units and a ReLU activation, followed a by an output layer of one sigmoid unit. Predictions are taken on a threshold basis, ie for outputs lower than 0.5, we predict 0 (died); and for ouputs higher we predict 1 (survived). The final architecture for this network was determined through trial and error experiments. For training, binary cross-entropy was used as a loss function, and the Adam optimiser was used with a learning rate of 0.00005 and weight decay of 0.002. These hyper-parameters were also selected through experiments.

## Results
The results can be found in the `log.txt` file.

What we have observed is an improved performance on the dataset as compared with other methods from the Scikit-learn package. The results from the Scikit-learn methods are presented below.

|Method|Accuracy score|
|---|---|
|Naive Bayes|0.71|
|Log. Regression|0.80|
|Random Forest|**0.83**|

As shown above in bold, the Random Forrest classifier was the most effective of the methods used. Because this is a simple binary classification task, no more sophisticated metrics for success are required than accuracy.

The neural network approach was able to outperform these methods; it is true that some variations (in terms of hyperparamters and architecture) of the neural network approach would actually underfperform some or all of the above results. With the correct configuration however, significant performance was acheived that we could not replicate on the Scikit-learn methods. Below is the final result for the neural network approach:

|Method|Accuracy score|
|---|---|
|Neural Network|**0.87**|

## The data
The data consists of the following columns, including the top 5 rows so as to exemplify the value formats:

![Dataframe columns](Images/head.png)

Here, `ParCh` indicates the number of parents/children that accompanied the passenger. `SibSp` indicates the number of sibllings/spouses. `Embarked` refers to which port the passenger embarked on. The `Pclass` column represents ticket classes, ie "first class", "second class" and "third class".

Below shows the data types of the columns, and the number of entries for that column. What we see is that `Age` and `Cabin` have some missing values:

![Dataframe info](Images/info.png)

Below gives a discription of numerical data in the dataframe:

![Dataframe description](Images/describe.png)

The mean average for the `Survived` feature is 0.38. Given this is a binary value, with 1 indicating survival, we can infer a 38% chance of survival for passengers in the training set. Collumns such as `Age` and `Fare` have a high standard deviation, which suggests they may need to be normalised. We can also see that 75% of passengers paid less than $31, despite the mean being $32.20 - this is because of extremely high values, such as the maximum ticket fare of $512.33. This further suggests this column can be normalised.

Below are histograms for some numerical values. The `Age` column shows a fairly normal distribution, but the others do not, indicating that some feature engineering maybe of use to normalise them.

![Age](Images/age.png)

![Fare](Images/fare.png)

![Parents/Children](Images/parch.png)

![Siblings/Spouse](Images/sibsp.png)

For non-numerical values, we can get counts of appearances of a given value. For example, with the `Embarked` column we can the number of passengers who embarked at each of the locations, and represent this distribution with a box plot. Some non-numerical types do not represent categories; for example, `Name` and `Cabin` have unique values, and therefore have not been illustrated below.

![Survived](Images/survived.png)

![Sex](Images/sex.png)

![Ticket class](Images/pclass.png)

![Embarking location](Images/embarked.png)

The locations corresponding the embarking categories are `S = Southampton`, `Q = Queensland` & `C = Cherbourg`.

At this stage we should understand more about the structure and shape of the training dataset. To take this understanding further, we can examine the relationships between the different properties of the samples; we are particularly interested in the independant variables' relationships to the `Survived` property.

Below is a heatmap that illustrates the correlation between each property. Using the legend on the right, squares closer to one (light) have a higher positive correlation, whilst squares closer to -0.5 (dark) have a higher negative correlation. A zero indicates no relationship. In order to include the `Sex` column, we changed `female = 1` and `male = 0`.

![Data heatmap](Images/heatmap.png)

According to the heatmap, there is a significant negative correlation between ticket class and survival; this shows that being in first class (denoted by a value of 1) makes a passenger more likely to survive than one in third class (denoted by a value of 3). We can also observe a positive correlation between the cost of the ticket and the survival rate, though this may be a mirror of the pattern observed between `Pclass` and `Survived`. We also see that females have a higher chance of surviving than males.

If we take a pivot table of the numerical data, we can see the average values of those who survived and those who died:

![Numerical data](Images/pivotNum.png)

The average ticket price of survivors is far higher than that of non-survivors, which indicates that `Fare` is a particularly important column for our purposes. Similarly, we can see that there is potential value in modelling the `Parch` and `SibSp` columns.

For categorical data, we can use value counts to illustrate how many of each category survived and how many did not.

![Embarked](Images/pivotEmb.png)

![Ticket class](Images/pivotCla.png)

![Sex](Images/pivotSex.png)

For the `Embarked` column, we can see passengers embarking from Cherbourg have a higher survival rate than those from Southhampton, with Queensland falling in between. We see from the `Sex` column that vast majority of females survived whilst the vast majority of males did not, meaning that this data maybe particularly important for our prediction model. We also see the correlation between ticket class and survival illustrated again, suggesting that this also is an important feature for our model.

Finally, we can explore the `Cabin` column. The given values for this column are not useful as the Cabin IDs are unique. However, some passenger have access to multiple cabins under their name. Below is a pivot table generated on the number of cabins under a passengers name:

![Cabin count](Images/cabinCount.png)

In addition, we can take the first letter of the cabin to bbe a given category (where `n` indicates null value):

![Cabin letters](Images/cabinLetter.png)

Both of these tables indicate that there may be relevent information in the distribution of passengers into these categories.

## Data processing

In order to process this data more effectively, some feature engineering and data preparation were conducted before the dataset was exposed to our models. The following is a breif account of the actions taken, and motivations for them.

### Impute the data

The collumns 'Age' and 'Cabin' had some missing values. To rectify this for the age collumn, the data was imputed with the mean vallue for age. This means every missing data point was replaced with the mean age value. This is appropriate because there is only a small fraction of this data missing; if a more significant number of value in the age collumn were missing, it may not be appropriate to use the mean of the data points we *do* have, because these data may not be representative. For this reason, and because it is of a special type, the 'Cabin' column was dropped all together.

### Feature engineering on Cabin column

Although the column itself was dropped, the data in the cabin column have still been put to use. In the 'Data' section of this document, we note that there are posible impacts on survival chance of 1) the number of cabins registered to the passengers name, and 2) the first letter of the cabin number. To incorporate this information, the number of cabins has been added as a column, and the first letters have been made into categories for which each passenger has a one-hot encoded value. Despite the large number of missing values, we found that the inclusion of these data improved performance on the network.

### One-hot encode the data

For the categorical columns (cabin letters, sex, embarking location) one-hot encodings were produced. This meant that the columns themselves were removed, and replaced with columns that reflect their possible values, ie the 'Sex' column would be dropped, and new columns 'Male' and 'Female' would be added. For any given row, all values for these new collumns are `0`, except in the column that represents the value associated with that passenger which is a `1`.

### Normalise

Finally the data is normalised, meaning it is centered around 0 and scaled to a real number value between -1 and 1. This makes the convergence of our models faster and efficient, and makes the model more effective at generalising to new data.

## References & resources

Kaggle - https://www.kaggle.com

Ken Jee - https://www.youtube.com/watch?v=I3FBJdiExcg

NeuralNine - https://www.youtube.com/watch?v=fATVVQfFyU0

Pytorch - https://pytorch.org

