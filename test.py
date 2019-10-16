import feature_engineering as fe
import neural_network as nn
import random_forest as rf
from keras.models import load_model

FILEPATH_X = "multilabel/X.csv"
FILEPATH_Y = "multilabel/Y_Type.csv"
TEST_SIZE = 0.05

df = fe.read_csv(FILEPATH_X)

df = fe.one_hot_encode(df, ['Formation'])
df = fe.one_hot_encode(df, ['YardLineDirection'])
df = fe.one_hot_encode(df, ['OffenseTeam'])
df = fe.one_hot_encode(df, ['DefenseTeam'])

df = fe.run_pca(df)


labels = fe.read_csv(FILEPATH_Y)

labels = fe.label_encode(labels)


X_train, X_test, y_train, y_test = fe.split_train_test(df, labels, TEST_SIZE)

num_features = fe.get_num_features(X_train)

classifier = nn.build_nn(num_features)
classifier = nn.compile_nn(classifier)
nn.train_nn(classifier, X_train, y_train)

#classifier.save('play_predictor.h5')


#from keras.models import load_model
#import neural_network as nn

#classifier = load_model('play_predictor.h5')
#cm, y_pred = nn.test_nn(classifier, X_test, y_test)
#print(cm)







