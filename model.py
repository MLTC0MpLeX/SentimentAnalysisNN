from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Input
from config import MAX_FEATURES

def build_model(input_dim, num_classes):
    model = models.Sequential()
    model.add(Input(shape=(input_dim,), sparse=True))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
