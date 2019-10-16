from keras.models import load_model
import neural_network as nn

classifier = load_model('play_predictor.h5')
cm, y_pred = nn.test_nn(classifier, X_test, y_test)
print(cm)
