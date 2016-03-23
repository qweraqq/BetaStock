from keras.models import model_from_json
from Preprocess import Preprocessor

model = model_from_json(open('ca.model.json').read())
model.load_weights('ca.model.weights.h5')
preprocessor = Preprocessor()
X1, X2, y = preprocessor.readNewsFromFile('201501news.txt', max_len=200)
y_predict = model.predict([X2, X1],  batch_size=1)

for i,v in enumerate(y):
    print v, y_predict[i]