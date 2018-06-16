from keras.layers import LeakyReLU as ass
from keras.layers import PReLU as asss
from keras.layers import ELU as assss
from keras.layers import ThresholdedReLU as asssss
from keras.models import Sequential
from keras.layers import Dense, Activation
from util import get_normalized_data, y2indicator

import matplotlib.pyplot as plt

X, Y = get_normalized_data()

N, D = X.shape
K = len(set(Y))

Y = y2indicator(Y)

model = Sequential()

model.add(Dense(units=500, input_dim=D))
#model.add(Activation('selu'))
model.add(asssss(theta=1.0))
model.add(Dense(units=300))
model.add(asssss(theta=1.0))
#model.add(Activation('selu'))
model.add(Dense(units=K))
model.add(Activation('softmax'))


model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


r = model.fit(X, Y, validation_split=0.33, epochs=15, batch_size=32)
print("Returned:", r)

print(r.history.keys())

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()
