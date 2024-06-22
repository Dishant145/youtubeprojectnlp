import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import pickle
import sys
from src.nlpyoutube.logger import logging
from src.nlpyoutube.exception import CustomException

class ModelTrainer:
    def __init__(self):
        self.max_features = 7000
        self.maxlen = 150
        self.embedding_dims = 150
        self.batch_size = 32
        self.epochs = 15
        self.tokenizer = None

    def preprocess(self, text):
        stop = stopwords.words('english')
        punc = list(punctuation)
        bad_tokens = stop + punc
        lemma = WordNetLemmatizer()
        tokens = word_tokenize(text)
        word_tokens = [t for t in tokens if t.isalpha()]
        clean_token = [lemma.lemmatize(t.lower()) for t in word_tokens if t not in bad_tokens]
        return ' '.join(clean_token)

    def train_model(self, data_path, tokenizer_path, model_path):
        try:
            data = pd.read_csv(data_path)
            data.dropna(subset=['Comment'], inplace=True)
            data['text'] = data['Comment'].apply(self.preprocess)
            data.dropna(subset=['text'], inplace=True)

            self.tokenizer = Tokenizer(num_words=self.max_features)
            self.tokenizer.fit_on_texts(data['text'])
            sequences = self.tokenizer.texts_to_sequences(data['text'])
            padded_sequences = pad_sequences(sequences, maxlen=self.maxlen)

            with open(tokenizer_path, 'wb') as file:
                pickle.dump(self.tokenizer, file)

            labels = data['sentiment']
            X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

            model = Sequential()
            model.add(Embedding(self.max_features, self.embedding_dims))
            model.add(Dropout(0.3))
            model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(3, activation='softmax', kernel_regularizer=l2(0.01)))

            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.3, callbacks=[early_stopping], verbose=2)
            model.save(model_path)

            loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
            logger.info(f'Test Accuracy: {accuracy}')
        except Exception as e:
            raise CustomException(e,sys)

