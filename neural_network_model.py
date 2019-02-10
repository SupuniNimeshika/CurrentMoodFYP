from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Input, Dense,Dropout, Embedding,LSTM, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import data_extract as data_source
import pickle
from text_processing import pre_process
from ploting_graph import plot_history
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

(unprocessed_texts, label) = data_source.get_data()
text =[]
for unprocessed_text in unprocessed_texts:
    text.append(pre_process(unprocessed_text))

MAX_LENGTH =60
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
post_seq =tokenizer.texts_to_sequences(text)
post_seq_padded =pad_sequences(post_seq,maxlen=MAX_LENGTH,padding='post')

X_train, X_test, y_train, y_test =train_test_split(post_seq_padded,label, test_size=0.3)
print(to_categorical(y_train))
vocab_size =len(tokenizer.word_index)+1
print(vocab_size)

inputs = Input(shape=(MAX_LENGTH, ))
embedding_layer =Embedding(vocab_size,128,input_length=MAX_LENGTH)(inputs)
x = Flatten()(embedding_layer)
x = Dense(32, activation='relu')(x)

predictions = Dense(4,activation='softmax')(x)
model =Model(inputs=[inputs],outputs=predictions)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
model.summary()
filepath ="model/NN-model.hdf5"

checkpointer = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode ='max')

history =model.fit(X_train,
                   to_categorical(y_train),
                   batch_size=64,
                   verbose=1,
                   validation_split=0.25,
                   shuffle=True,
                   epochs=25,
                   callbacks=[checkpointer])

predictions = model.predict(np.array(X_test))
prediction=predictions.argmax(axis=1)
label=np.array(to_categorical(y_test)).argmax(axis=1)
cm=confusion_matrix(label,prediction)
print(cm)
cr=classification_report(label,prediction,target_names=['happy','sad','calm','angry'])
print(cr)
acc=accuracy_score(label,prediction)
print(acc)

with open("model.pickle","wb") as f:
   pickle.dump(tokenizer,f)


loss, accuracy = model.evaluate(X_train, to_categorical(y_train), verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, to_categorical(y_test), verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)
