import pandas as pd
import numpy as np
import seaborn as sns
from Titanic import config

def load_csv():
    train = pd.read_csv(config.TRAIN_FILE)
    test = pd.read_csv(config.TEST_FILE)
    return train, test

def dummies(col, train, test):
    '''
    The function can convert labels into numbers

    :param col: Attributes
    :param train: train dataframe
    :param test: test dataframe
    :return: new train DF and new test DF
    '''
    train_dum = pd.get_dummies(train[col])
    test_dum = pd.get_dummies(test[col])
    train = pd.concat([train, train_dum], axis=1)
    test = pd.concat([test, test_dum], axis=1)
    train.drop(col, axis=1, inplace=True)
    test.drop(col, axis=1, inplace=True)
    return train, test

def small15(row):
    result = 0.0
    if row < 15:
        result = 1.0
    return result

def big15(row):
    result = 0.0
    if row >= 15 and row < 30:
        result = 1.0
    return result

def process(train, test):
    # Delete <PassengerId> <Name> <Ticket>
    dropping = ['PassengerId', 'Name', 'Ticket']
    train.drop(dropping, axis=1, inplace=True)
    test.drop(dropping, axis=1, inplace=True)

    # process <Pclass>
    # print(train.Pclass.value_counts())
    sns.factorplot('Pclass', 'Survived', data=train, order=[1, 2, 3])
    train, test = dummies('Pclass', train, test)
    # train.info()

    # process <Sex>
    # print(train.Sex.value_counts(dropna=False))
    sns.factorplot('Sex', 'Survived', data=train)
    train, test = dummies('Sex', train, test)
    train.drop('male', axis=1, inplace=True)
    test.drop('male', axis=1, inplace=True)

    # process <Age>
    nan_num = len(train[train['Age'].isnull()])
    age_mean = train['Age'].mean()
    age_std = train['Age'].std()
    filling = np.random.randint(age_mean - age_std, age_mean + age_std, size=nan_num)
    train['Age'][train['Age'].isnull() == True] = filling

    nan_num = len(test[test['Age'].isnull()])
    age_num = test['Age'].mean()
    age_std = test['Age'].std()
    filling = np.random.randint(age_mean - age_std, age_mean + age_std, size=nan_num)
    test['Age'][test['Age'].isnull() == True] = filling

    s = sns.FacetGrid(train, hue='Survived', aspect=2)
    s.map(sns.kdeplot, 'Age', shade=True)
    s.set(xlim=(0, train['Age'].max()))
    s.add_legend()

    train['under15'] = train['Age'].apply(small15)
    train['up15'] = train['Age'].apply(big15)
    test['under15'] = test['Age'].apply(small15)
    test['up15'] = test['Age'].apply(big15)

    train.drop('Age', axis=1, inplace=True)
    test.drop('Age', axis=1, inplace=True)

    # process <SibSp> & <Parch>
    # print(train.SibSp.value_counts(dropna=False))
    # print(train.Parch.value_counts(dropna=False))
    sns.factorplot('SibSp', 'Survived', data=train)
    sns.factorplot('Parch', 'Survived', data=train)

    train['family'] = train['SibSp'] + train['Parch']
    test['family'] = test['SibSp'] + test['Parch']

    train.drop(['SibSp', 'Parch'], axis=1, inplace=True)
    test.drop(['SibSp', 'Parch'], axis=1, inplace=True)

    # process <Fare>
    # print(train.Fare.isnull().sum())
    # print(test.Fare.isnull().sum())
    test['Fare'].fillna(test['Fare'].median(), inplace=True)

    # process <Cabin>
    train.drop('Cabin', axis=1, inplace=True)
    test.drop('Cabin', axis=1, inplace=True)

    # process <Embarked>
    print(train.Embarked.isnull().sum())
    print(test.Embarked.isnull().sum())
    print(train.Embarked.value_counts(dropna=False))
    train['Embarked'].fillna('S', inplace=True)
    sns.factorplot('Embarked', 'Survived', data=train, size=5)
    train, test = dummies('Embarked', train, test)
    train.drop(['S', 'Q'], axis=1, inplace=True)
    test.drop(['S', 'Q'], axis=1, inplace=True)
    return train, test


