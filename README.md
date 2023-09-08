import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load your dataset, where X contains audio features and y contains emotion labels.
# Make sure to preprocess your dataset accordingly.

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction (you may want to use more advanced features)
def extract_features(audio_file):
    # Use librosa to extract features like MFCCs, chroma, etc.
    audio_data, sr = librosa.load(audio_file)
    features = librosa.feature.mfcc(y=audio_data, sr=sr)
    return np.mean(features, axis=1)

X_train_features = [extract_features(audio_file) for audio_file in X_train]
X_test_features = [extract_features(audio_file) for audio_file in X_test]

# Train a simple classifier (SVM in this case)
classifier = SVC(kernel='linear')
classifier.fit(X_train_features, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_features)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
