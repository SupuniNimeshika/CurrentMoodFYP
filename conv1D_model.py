from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, Dense, Embedding,GlobalMaxPooling1D
import data_extract as data_source
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import to_categorical
import pickle


(text, label) = data_source.get_data()
embedding_dim = 100

MAX_LENGTH =100

tokenizer = Tokenizer()
with open("model.pickle","rb") as f:
    tokenizer = pickle.load(f)
post_seq =tokenizer.texts_to_sequences(text)
post_seq_padded =pad_sequences(post_seq,maxlen=MAX_LENGTH,padding='post')

X_train, X_test, y_train, y_test =train_test_split(post_seq_padded,label, test_size=0.25)
vocab_size =len(tokenizer.word_index)+1

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=MAX_LENGTH))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, to_categorical(y_train),
                    epochs=20,
                    verbose=False,
                    validation_data=(X_test, to_categorical(y_test)),
                    batch_size=64)

model.save('model/CNN-model.hdf5')


loss, accuracy = model.evaluate(X_train, to_categorical(y_train), verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, to_categorical(y_test), verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
# plot_history(history)

