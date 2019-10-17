"""
Name: Karan Pankaj makhija and Jeet thakur
Version: Python 2.7
Title: LSTM using tensorflow and kernas
"""
from keras.models import Sequential
from keras import layers
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

#Making the learning rate as 0.01
opt = SGD(lr=0.01)
plt.style.use('ggplot')



def history_plot(history):
    """
    Plot the history of each point
    :param history: Total history
    :return: None
    """
    #History has 4 parameters, Accuracy and Loss for training, Accuracy and Loss for validation
    accuracy = history.history['acc']
    loss = history.history['loss']
    test_accuracy = history.history['val_acc']
    test_loss = history.history['val_loss']

    x = range(1, len(accuracy) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, accuracy, 'b', label='Training acc')
    plt.plot(x, test_accuracy, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, test_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#Reading the file
data = pd.read_json('Final1.json')

X = data.iloc[:,1].values
Y = data.iloc[:,0].values
labelencoder_y_1 = LabelEncoder()
Y = labelencoder_y_1.fit_transform(Y)
#Splitiing the data
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=0,shuffle=True)


#Coverting to feature vectors
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(X)
xtrain_tfidf =  tfidf_vect.transform(x_train)
xvalid_tfidf =  tfidf_vect.transform(x_test)

#Choosing the dimension as columns
input_dim = xtrain_tfidf.shape[1]
#Adding layers
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
print(model.summary())

#Fitting the data
history = model.fit(xtrain_tfidf, y_train,epochs=100,verbose=False,validation_data=(xvalid_tfidf, y_test),batch_size=10)
print(history)

#Evaluate the loss and accuracy on training data
loss, accuracy = model.evaluate(xtrain_tfidf, y_train, verbose=False)
print("Training Accuracy:",accuracy,loss)

#Evaluate the loss and accuracy on validation data
loss_test,accuracy_test = model.evaluate(xvalid_tfidf,y_test,verbose=False)
print("Testing Accuracy",accuracy_test)

#Plot the data
history_plot(history)

