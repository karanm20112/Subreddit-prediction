"""
Name: Karan Pankaj Makhija
Version: Python 2.7
Title : ROC for OnevsRest classifier SVM
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.utils.fixes import signature
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt



#Importing the JSON
data = pd.read_json('C:\\Users\\karan\\Desktop\\Final1.json')


X = data.iloc[:,1].values

Y = data.iloc[:,0].values

#Binarize the data. In 0 and 1 form
Y = label_binarize(Y,classes=[0,1,2])

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=0,shuffle=True)
#Convert it to feature Matrix
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(X)
xtrain_tfidf =  tfidf_vect.transform(x_train)
xvalid_tfidf =  tfidf_vect.transform(x_test)

#OneVSRestClassifier for SVM
svm1 = OneVsRestClassifier(LinearSVC(random_state=0))
prediction = svm1.fit(xtrain_tfidf, y_train).decision_function(xvalid_tfidf)

#Plot for everything
#It will be Class 0 vs Class 1 and Class 2
#Then Class 1 vs Class 2 and Class 0
#Then Class 2 vs class 0 and class 1
#plotting the entire the Precision and Recall graph on weighted Average
fpr = dict()
precision = dict()
recall = dict()
average_precision = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], prediction[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    precision[i],recall[i], _ = precision_recall_curve(y_test[:, i], prediction[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], prediction[:, i])

precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),prediction.ravel())
average_precision["micro"] = average_precision_score(y_test, prediction,average="micro")

print('Average precision score, micro-averaged of all the classes: {0:0.2f}'
      .format(average_precision["micro"]))

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
                 **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))


for i in range(3):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate '+'class:'+str(i))
    plt.ylabel('True Positive Rate '+'class'+str(i))
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()





