from tensorflow.keras import models
from data_loader import load_and_preprocess_data

def evaluate():
    _, X_test, _, y_test, vectorizer, label_encoder = load_and_preprocess_data()
    model = models.load_model('best_model.h5')

    # Evaluating the model on the test data
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test accuracy: {accuracy*100:.2f}%')

def predict(text):
    model = models.load_model('best_model.h5')
    text_transformed = models.vectorizer.transform([text]).toarray()
    predictions = model.predict(text_transformed)
    predicted_class = models.label_encoder.inverse_transform([predictions.argmax()])
    return predicted_class

if __name__ == "__main__":
    evaluate()
