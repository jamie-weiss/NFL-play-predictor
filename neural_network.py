from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

def build_nn(num_features):
	classifier = Sequential()
	#First Hidden Layer
	classifier.add(Dense(12,
						 activation='relu',
						 kernel_initializer='random_normal',
						 input_dim=num_features))
	#Second  Hidden Layer
	classifier.add(Dense(8,
						 activation='relu',
						 kernel_initializer='random_normal'))
	#Output Layer
	classifier.add(Dense(1,
						 activation='sigmoid',
						 kernel_initializer='random_normal'))
	return classifier

def compile_nn(classifier):
	classifier.compile(optimizer ='adam',
					   loss='binary_crossentropy',
					   metrics =['accuracy'])
	return classifier

def train_nn(classifier, X_train, Y_train):
	classifier.fit(X_train, Y_train, batch_size=10, epochs=20)

def test_nn(classifier, X_test, Y_test):
	Y_pred=classifier.predict(X_test)
	Y_pred =(Y_pred>0.5)
	cm = confusion_matrix(Y_test, Y_pred)
	return cm, Y_pred

def predict(classifier, X):
	Y_pred=classifier.predict(X)
	Y_pred =(Y_pred>0.5)
	return Y_pred