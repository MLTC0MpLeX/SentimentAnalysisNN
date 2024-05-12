from data_loader import load_and_preprocess_data
from model import build_model
from tensorflow.keras import callbacks
from config import EPOCHS, BATCH_SIZE

def train():
    X_train, X_test, y_train, y_test, vectorizer, label_encoder = load_and_preprocess_data()
    input_dim = X_train.shape[1]
    num_classes = y_train.shape[1]

    model = build_model(input_dim, num_classes)

    # Model checkpoint to save the best model
    checkpoint = callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

    # Training the model
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint]
    )

    return model, vectorizer, label_encoder

if __name__ == "__main__":
    train()
