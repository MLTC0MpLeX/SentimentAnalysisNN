import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras import utils
from sklearn.preprocessing import LabelEncoder
from config import DATA_PATH, MAX_FEATURES

def load_and_preprocess_data():
    # Load the dataset with correct column names
    # Columns: 0 - polarity, 1 - id, 2 - date, 3 - query, 4 - user, 5 - text
    columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
    data = pd.read_csv(DATA_PATH, encoding='ISO-8859-1', header=None, names=columns)

    # Filtering the necessary columns: 'text' for the tweet and 'polarity' for sentiment
    data = data[['text', 'polarity']]

    # Changing polarity to a binary format if necessary (0 = negative, 1 = positive)
    data['polarity'] = data['polarity'].replace(4, 1)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['polarity'], test_size=0.2, random_state=42)

    # Vectorizing the text data
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Encoding labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Convert labels to categorical format
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)

    return X_train, X_test, y_train, y_test, vectorizer, label_encoder
