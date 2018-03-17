import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame as df
import DataProcess
import initML

# clf: Classification
def modeling(clf, feature, target, kf):
    acc = cross_val_score(clf, feature, target, cv=kf)
    acc_lst.append(acc.mean())
    print(acc_lst)

accuracy = []

def ML(feature, target, kf):
    # LogisticRegression
    LR = LogisticRegression()
    modeling(LR, feature, target, kf)
    # RandomForest
    RF = RandomForestClassifier()
    modeling(RF, feature, target, kf)
    # SVM
    SVM = SVC()
    modeling(SVM, feature, target, kf)
    # KNN
    KNN = KNeighborsClassifier()
    modeling(KNN, feature, target, kf)

train, test = DataProcess.load_csv()
train, test = DataProcess.process(train, test)

train_ft = train.drop('Survived',axis=1)
train_y = train['Survived']
kf = KFold(n_splits=3,random_state=1)
acc_lst = []
ML(train_ft,train_y,kf)

# testing 2, lose young
train_ft_2=train.drop(['Survived','under15'],axis=1)
test_2 = test.drop('under15',axis=1)
train_ft.head()
kf = KFold(n_splits=3,random_state=1)
acc_lst=[]
ML(train_ft_2,train_y, kf)

#test3, lose young, c
train_ft_3=train.drop(['Survived','under15','C'],axis=1)
test_3 = test.drop(['under15','C'],axis=1)
train_ft.head()
kf = KFold(n_splits=3,random_state=1)
acc_lst = []
ML(train_ft_3,train_y,kf)

# test4, no FARE
train_ft_4=train.drop(['Survived','Fare'],axis=1)
test_4 = test.drop(['Fare'],axis=1)
train_ft.head()
kf = KFold(n_splits=3,random_state=1)
acc_lst = []
ML(train_ft_4,train_y, kf)

# test5, get rid of c
train_ft_5=train.drop(['Survived','C'],axis=1)
test_5 = test.drop('C',axis=1)
kf = KFold(n_splits=3,random_state=1)
acc_lst = []
ML(train_ft_5,train_y,kf)

# test6, lose Fare and young
train_ft_6 = train.drop(['Survived', 'Fare', 'under15'], axis=1)
test_6 = test.drop(['Fare', 'under15'], axis=1)
train_ft.head()
kf = KFold(n_splits=3, random_state=1)
acc_lst = []
ML(train_ft_6, train_y, kf)

# Testing

svc = SVC(C=0.5, kernel='sigmoid')

svc.fit(train_ft_4,train_y)
svc_pred = svc.predict(test_4)

print(svc.score(train_ft_4,train_y))


# Get Submission File
submission_test = pd.read_csv('data/test.csv')
submission = pd.DataFrame({"PassengerId":submission_test['PassengerId'],
                          "Survived":svc_pred})
submission.to_csv('data/kaggle_SVC.csv',index=False)

