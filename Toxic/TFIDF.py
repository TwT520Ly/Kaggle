import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
# print(train.head(5))
'''
id comment_test toxic severe_toxic obscene threat insult identity_hate
'''
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train_comment = train['comment_text']
test_comment = test['comment_text']
all_text = pd.concat([train_comment, test_comment])

# TF-IDF
word_vectorize = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    token_pattern='r\w{1,}',
    ngram_range=(1,1),
    strip_accents='unicode',
    max_features=10000,
    analyzer='word'
)
word_vectorize.fit(all_text)
train_word_feature = word_vectorize.transform(train_comment)
test_word_feature = word_vectorize.transform(test_comment)
'''
train_comment: (159571,)
train_word_feature: (159571, 10000)
test_comment: (153164)
test_word_feature: (153164, 10000)
'''

char_vectorize = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    token_pattern='r\w{1,}',
    ngram_range=(2,6),
    strip_accents='unicode',
    max_features=50000,
    analyzer='char'
)
char_vectorize.fit(all_text)
train_char_feature = char_vectorize.transform(train_comment)
test_char_feature = char_vectorize.transform(test_comment)

train_feature = hstack([train_word_feature, train_char_feature])
test_feature = hstack([test_word_feature, test_char_feature])

submission = pd.DataFrame.from_dict({'id', test['id']})
score = []
for class_name in class_names:
    train_target = train[class_name]
    clf = LogisticRegression(C=0.1, solver='sag')

    cv_score = np.mean(cross_val_score(clf, train_feature, train_target, cv=3, scoring='roc_auc'))
    score.append(cv_score)

    clf.fit(train_feature, train_target)
    submission[class_name] = clf.predict_proba(test_feature)[:, 1]
    print(submission[class_name].shape)