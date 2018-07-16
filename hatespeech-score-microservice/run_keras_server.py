# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'

# import the necessary packages
import importlib
import os
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import pickle
import flask
import io
import pandas as pd
from keras.preprocessing import text
import pickle

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend

def save_tokenizer():
    train = pd.read_csv('Train38k.csv')
    test = pd.read_csv('Test38k.csv')

    x_train = train["content"].fillna("fillna")
    x_test = test["content"].fillna("fillna")

    x_train = x_train.str.lower()
    x_test = x_test.str.lower()

    max_features = 100000
    maxlen = 150

    tok = text.Tokenizer(num_words=max_features, lower=True)

    tok.fit_on_texts(list(x_train) + list(x_test))

    with open('heise_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
def load_keras_model():
    global model
    K.clear_session()
    model = load_model("model_full.h5")

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned
    data = {"success": False}

    # ensure a comment was properly uploaded to our endpoint
    if flask.request.method == "POST":
        content = flask.request.get_json()
        if content:
            comment = content["comment"]
            with open('./heise_tokenizer.pickle', 'rb') as tok:
                tokenizer = pickle.load(tok)
                x_test = tokenizer.texts_to_sequences([comment])
                query = pad_sequences(x_test, maxlen=150)
                # model = load_model("model1_demo.h5")
                predictions = model.predict(query)

            print(predictions)
            data["prediction"] = predictions.item(0)
            data["success"] = True
            # K.clear_session()

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Flask starting server... "
           "please wait until server has fully started"))
    set_keras_backend("theano")
    save_tokenizer()
    load_keras_model()
    app.run(host='0.0.0.0')
