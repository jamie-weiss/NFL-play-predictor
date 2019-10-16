from sklearn.ensemble import RandomForestClassifier

def build_forest():
	clf = RandomForestClassifier(n_estimators=100,
								 max_depth=10,
								 random_state=0,
								 )
	return clf

def fit_forest(clf, X, y):
	clf.fit(X, y)
	return clf

def predict(clf, X):
	return clf.predict(X)

def predict_probs(clf, X):
	return clf.predict_proba(X)

def get_score(clf, test_X, true_Y):
	return clf.score(test_X, true_Y)