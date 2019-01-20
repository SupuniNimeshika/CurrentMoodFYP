from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Input, Dense,Dropout, Embedding,LSTM, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import data_extract as data_source
import pickle
from ploting_graph import plot_history

(text, label) = data_source.get_data()

''''
data['target'] = data.tags.astype('category').cat.codes
data['num_words'] = data.post.apply(lambda x : len(x.split()))
bins = [0,50,75, np.inf]
word_distribution = data.groupby('bins').size().reset_index().rename(columns={0:'counts'})
num_class = len(np.unique(data.tags.values))
y = data['target'].values
'''

MAX_LENGTH =100
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
post_seq =tokenizer.texts_to_sequences(text)
post_seq_padded =pad_sequences(post_seq,maxlen=MAX_LENGTH,padding='post')

X_train, X_test, y_train, y_test =train_test_split(post_seq_padded,label, test_size=0.05)
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
filepath ="NN-model.hdf5"

checkpointer = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode ='max')

history =model.fit(X_train,
                   to_categorical(y_train),
                   batch_size=64,
                   verbose=1,
                   validation_split=0.15,
                   shuffle=True,
                   epochs=10,
                   callbacks=[checkpointer])

# predicted = model.predict(X_test)
# predicted =np.argmax(predicted,axis=1)
# accuracy_score(y_test,predicted)

with open("model.pickle","wb") as f:
   pickle.dump(tokenizer,f)
#
# model.save('model.hdf5')

'''
from keras.models import load_model
model = load_model("model.hdf5")

with open("model.pickle","rb") as f:
    tokenizer = pickle.load(f)

# Testing with new data
newtexts = ["Your new data", "Everything looks so stressed","sad"]

sequences = tokenizer.text_to_sequences(newtexts)
data = pad_sequences(sequences,maxlen=500)

prediction =model.predict(data)
print(predictions)
prob = predictions.argmax(axis=1)
print(prob)
'''

#model = load_model("model.hdf5")

# MAX_LENGTH =500
# tokenizer = Tokenizer()
# # Testing with new data
# text=['happy']
#
# sequences = tokenizer.texts_to_sequences(text)
# print(sequences)
# data = pad_sequences(sequences,maxlen=500)
# print(X_test)
# prediction =model.predict(X_test)
# print(prediction)

loss, accuracy = model.evaluate(X_train, to_categorical(y_train), verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, to_categorical(y_test), verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)
