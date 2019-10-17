"""
Name: Karan pankaj Makhija and Jeet Thakur
Version: Python 2.7
Title : Multiclass_Classification with Feature Engineering
"""
from sklearn import preprocessing, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import ensemble
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn import metrics
from nltk.stem import PorterStemmer

#Finding the stem of the words
ps = PorterStemmer()
my_stopwords = stopwords.words('english')
data = pd.read_json('Final1.json')

#Converting to lower case
data.iloc[:,1] = data.iloc[:,1].apply(lambda x:" ".join(x.lower() for x in x.split()))
#Removing Punctuation
data.iloc[:, 1] = data.iloc[:,1].str.replace('[^\w\s]','')
#Removing StopWords
data.iloc[:, 1] = data.iloc[:,1].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
data.iloc[:, 1] = data.iloc[:,1].apply(lambda x: " ".join(x for x in x.split() if x not in my_stopwords))

X = data.iloc[:,1].values
Y = data.iloc[:,0].values
#Splitting the data
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=0,shuffle=True)
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

#Feature Scaling with Count Vectorizer
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(X)
xtrain_count =  count_vect.transform(x_train)
xvalid_count =   count_vect.transform(x_test)

#Feature Scaling with TFIFD vectorizer
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(X)
xtrain_tfidf =  tfidf_vect.transform(x_train)
xvalid_tfidf =  tfidf_vect.transform(x_test)




def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    """
    Fit the model and then Prediction on test data
    :param classifier: Model Name
    :param feature_vector_train: Training input data
    :param label: Training output label
    :param feature_vector_valid: Testing input data
    :return: Accuracy score
    """
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    return metrics.accuracy_score(predictions, y_test)

accuracy = train_model(svm.SVC(kernel='linear'), xtrain_tfidf, y_train, xvalid_tfidf)
print "SVM, WordLevel TF-IDF: Accuracy:", accuracy*100
print("\n")
accuracy = train_model(svm.SVC(kernel='linear'), xtrain_count, y_train, xvalid_count)
print "SVM, CountVector: Accuracy:", accuracy*100
print("\n")
accuracy= train_model(ensemble.RandomForestClassifier(n_estimators=100), xtrain_tfidf, y_train, xvalid_tfidf)
print "RF, Wordlevel TF-IDF: Accuracy:", accuracy*100
print("\n")
accuracy= train_model(ensemble.RandomForestClassifier(n_estimators=100), xtrain_count, y_train, xvalid_count)
print "RF, CountVector: Accuracy:", accuracy*100
print("\n")


