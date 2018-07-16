import pandas as pd
from keras.preprocessing import text
import pickle

data_path = 'D:/Studium/EDM/'

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