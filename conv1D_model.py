from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, Dense, Embedding, GlobalMaxPooling1D, Dropout
import data_extract as data_source
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import to_categorical
import pickle
from text_processing import pre_process
from ploting_graph import plot_history
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# “ ’ ‘ ” #

(unprocessed_texts, label) = data_source.get_data()
text =[]
for unprocessed_text in unprocessed_texts:
    text.append(pre_process(unprocessed_text))
    # print(pre_process(unprocessed_text))
embedding_dim = 60

MAX_LENGTH =60
tokenizer = Tokenizer()
with open("model.pickle","rb") as f:
    tokenizer = pickle.load(f)
post_seq =tokenizer.texts_to_sequences(text)
post_seq_padded =pad_sequences(post_seq,maxlen=MAX_LENGTH,padding='post')

X_train, X_test, y_train, y_test =train_test_split(post_seq_padded,label, test_size=0.5)
vocab_size =len(tokenizer.word_index)+1

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=MAX_LENGTH))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
filepath ="model/CNN-model.hdf5"

checkpointer = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode ='max')

history = model.fit(X_train, to_categorical(y_train),
                    epochs=25,
                    verbose=True,
                    validation_data=(X_test, to_categorical(y_test)),
                    batch_size=64,
                    callbacks = [checkpointer])

# model.save('model/CNN-model.hdf5')
predictions = model.predict(np.array(X_test))
prediction=predictions.argmax(axis=1)
label=np.array(to_categorical(y_test)).argmax(axis=1)
cm=confusion_matrix(label,prediction)
print(cm)
cr=classification_report(label,prediction,target_names=['happy','sad','calm','angry'])
print(cr)
acc=accuracy_score(label,prediction)
print(acc)


loss, accuracy = model.evaluate(X_train, to_categorical(y_train), verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, to_categorical(y_test), verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)

