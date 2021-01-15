import pandas as pd
import numpy as np
from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *

df = pd.read_csv('spam.csv', encoding="latin-1")
df.head()

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df['SMS'] = df['v2']
df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
df.drop(['v1', 'v2'], axis=1, inplace=True)
train_data = df[:4400]
test_data = df[4400:]
df.head()

def perform(classifiers, vectorizers, train_data, test_data):
    max_score = 0
    max_name = 0
    for classifier in classifiers:
        for vectorizer in vectorizers:
        
            # train
            vectorize_text = vectorizer.fit_transform(train_data.SMS)
            classifier.fit(vectorize_text, train_data.label)

            # score
            vectorize_text = vectorizer.transform(test_data.SMS)
            score = classifier.score(vectorize_text, test_data.label)
            name = classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__ 
            print(name, score)
        if score > max_score:
            max_score = score
            max_name = name
    print ('===========================================')
    print ('===========================================')
    print (max_name, max_score)
    print ('===========================================')
    print ('===========================================')


classifiers = [
        BernoulliNB(),
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        AdaBoostClassifier(),
        BaggingClassifier(),
        ExtraTreesClassifier(),
        GradientBoostingClassifier(),
        DecisionTreeClassifier(),
        CalibratedClassifierCV(),
        DummyClassifier(),
        PassiveAggressiveClassifier(),
        RidgeClassifier(),
        RidgeClassifierCV(),
        SGDClassifier(),
        OneVsRestClassifier(SVC(kernel='linear')),
        OneVsRestClassifier(LogisticRegression()),
        KNeighborsClassifier()
    ]

vectorizers = [
        CountVectorizer(),
        TfidfVectorizer(),
        HashingVectorizer()
    ]


perform(
    classifiers,
    vectorizers,
    train_data,
    test_data
)


Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(train_data.SMS)
Classifier.fit(vectorize_text, train_data.label)


SMS = ' won a 1 week FREE membership in our $100,000 Prize Jackpot! Txt the word: C'
vectorize_message = Vectorizer.transform([SMS])
predict = Classifier.predict(vectorize_message)[0]



if predict == 0:
    print ('ham')
else:
    print ('spam')