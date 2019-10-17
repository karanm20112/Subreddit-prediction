"""
Name : Karan Makhija and Jeet Thakur
Version: Python 2.7
Title: Learning with their ROC curves
"""
#All the required imports
from sklearn import preprocessing, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import ensemble
import pandas as pd
from sklearn.model_selection import train_test_split,learning_curve
from nltk.corpus import stopwords
from sklearn import metrics
import matplotlib.pyplot as plt

#Defining Stop Words
my_stopwords = stopwords.words('english')
data = pd.read_json('Final1.json')
#Taking only first two reddits for classification
data = data[0:700]
#Shuffling the data
data = data.sample(frac=1)
#Converting to lower case
data.iloc[:,1] = data.iloc[:,1].apply(lambda x:" ".join(x.lower() for x in x.split()))
#Removing Punctuation
data.iloc[:, 1] = data.iloc[:,1].str.replace('[^\w\s]','')
#Removing StopWords
data.iloc[:, 1] = data.iloc[:,1].apply(lambda x: " ".join(x for x in x.split() if x not in my_stopwords))
#Body of subreddit
X = data.iloc[:,1].values
#Target value of that subreddit
Y = data.iloc[:,0].values
#Splitting the data
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=0,shuffle=True)
#Encoding the Target variables if it is not number.
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

#Converting it to feature matrix which does analysis on word with max features of 5000
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(X)
xtrain_tfidf =  tfidf_vect.transform(x_train)
xvalid_tfidf =  tfidf_vect.transform(x_test)

def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    """

    :param classifier: Classifier name
    :param feature_vector_train: Input training data
    :param label: Output training data
    :param feature_vector_valid: Input testing data
    :return: The various metrics based on prediction
    """
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    return metrics.accuracy_score(predictions, y_test),predictions,metrics.recall_score(predictions, y_test),\
           metrics.precision_score(predictions,y_test),metrics.confusion_matrix(predictions, y_test)


#Training the data on SVM model with a linear kernal
accuracy, prediction,recall, precision,confunsion_matrix = train_model(svm.SVC(kernel='linear'), xtrain_tfidf, y_train, xvalid_tfidf)
print "SVM, WordLevel TF-IDF: Accuracy:", accuracy*100,\
    " Recall:",recall*100," Precision:",precision*100,"\n Confusion Matrix:\n",confunsion_matrix

#Plotting the ROC curves for the given output. fpr: false positive rate, tpr: True positive rate
fpr, tpr, threshold = metrics.roc_curve(y_test, prediction)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic with SVM')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#Training the data on Random Forest model with n_estimators as 100
accuracy,prediction,recall,precision,confunsion_matrix = train_model(ensemble.RandomForestClassifier(n_estimators=100),
                                                                     xtrain_tfidf, y_train, xvalid_tfidf)
print "RF, WordLevel TF-IDF: Accuracy:", accuracy*100,\
    " Recall:",recall*100," Precision:",precision*100,"\n Confusion Matrix:\n",confunsion_matrix

#Plotting the ROC curve with Random Forest Model
fpr1, tpr1, threshold1 = metrics.roc_curve(y_test, prediction)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic with RandomForest')
plt.plot(fpr1, tpr1, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()










