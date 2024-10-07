from preprocess import *
from keras.layers import Dense
from keras.models import  Sequential
import keras.activations,keras.losses
from keras.callbacks import EarlyStopping


callback = EarlyStopping(monitor='val_accuracy',
                                  mode="max", 
                                  patience=3,
                                  restore_best_weights=True)

model=Sequential()
model.add(Dense(x.shape[1],input_dim=x.shape[1],activation=keras.activations.relu))
model.add(Dense(32,activation=keras.activations.relu))
model.add(Dense(64,activation=keras.activations.relu))
model.add(Dense(128,activation=keras.activations.relu))
model.add(Dense(256,activation=keras.activations.relu))
model.add(Dense(512,activation=keras.activations.relu))
model.add(Dense(Y.shape[1],activation=keras.activations.softmax))

model.summary()
model.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])



hist=model.fit(x_tran,y_tran,epochs=3, verbose = 1, validation_data =(x_tst, y_tst),callbacks = [callback])
plt.plot(hist.history['accuracy'], label='training accuracy', marker='o', color='red')
plt.plot(hist.history['loss'], label='loss', marker='o', color='darkblue')
plt.title('Training Vs  Validation accuracy with adam optimizer')
plt.legend()
plt.show()


test_loss, test_accuracy = model.evaluate(x_tst, y_tst)

print(f'Test Accuracy: {test_accuracy}')
print(f'Test Loss: {test_loss}')


model.save('pretranied_model/ptm.h5')