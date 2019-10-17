"""
Name : Karan Makhija and Jeet Thakur
Version: Python 2.7
Title: Learning curve
"""
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import ensemble
import pandas as pd
from sklearn.model_selection import learning_curve
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np

#CLeaning the data
my_stopwords = stopwords.words('english')
data = pd.read_json('Final1.json')
data = data[0:700]
data = data.sample(frac=1)
#Converting to lower case
data.iloc[:,1] = data.iloc[:,1].apply(lambda x:" ".join(x.lower() for x in x.split()))
#Removing Punctuation
data.iloc[:, 1] = data.iloc[:,1].str.replace('[^\w\s]','')
#Removing StopWords
data.iloc[:, 1] = data.iloc[:,1].apply(lambda x: " ".join(x for x in x.split() if x not in my_stopwords))
X = data.iloc[:,1].values
Y = data.iloc[:,0].values

#Converting it to feature vector
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(X)
X = tfidf_vect.transform(X)

#Creating learning curve with Randomforest model
train_sizes, train_scores, test_scores = learning_curve(ensemble.RandomForestClassifier(),X,Y,cv=10,scoring='accuracy',
                                                        train_sizes= np.linspace(0.01,1.0,75))
#Calculating mean and standard deviation.

t_mean = np.mean(train_scores, axis=1)
t_std = np.std(train_scores, axis=1)
add1 = t_mean+t_std
sub1 = t_mean-t_std
te_mean = np.mean(test_scores, axis=1)
te_std = np.std(test_scores, axis=1)
add2 = te_mean+te_std
sub2 = te_mean-te_std
#Plotting it
plt.plot(train_sizes,t_mean,'--',color= "#111111",label="Training Score")
plt.plot(train_sizes,te_mean,color= "#111111",label="Testing Score")
#Filling the error
plt.fill_between(train_sizes,add1,sub1,color="#808080")
plt.fill_between(train_sizes,add2,sub2,color="#808080")

plt.title("Learning curve for RandomForest")
plt.xlabel("Training set size")
plt.ylabel("Testing set size")
plt.show()
#Creating a learning curve for SVM
train_sizes, train_scores, test_scores = learning_curve(svm.SVC(kernel='linear'),X,Y,cv=10,scoring='accuracy',
                                                        train_sizes= np.linspace(0.01,1.0,75),shuffle=True)
#calculating for mean and standard deviation
t_mean = np.mean(train_scores, axis=1)
t_std = np.std(train_scores, axis=1)
add1 = t_mean+t_std
sub1 = t_mean-t_std
te_mean = np.mean(test_scores, axis=1)
te_std = np.std(test_scores, axis=1)
add2 = te_mean+te_std
sub2 = te_mean-te_std
#plotting it
plt.plot(train_sizes,t_mean,'--',color= "#111111",label="Training Score")
plt.plot(train_sizes,te_mean,color= "#111111",label="Testing Score")
#Filling the error
plt.fill_between(train_sizes,add1,sub1,color="#808080")
plt.fill_between(train_sizes,add2,sub2,color="#808080")

plt.title("Learning curve for SVM")
plt.xlabel("Training set size")
plt.ylabel("Testing set size")
plt.show()

