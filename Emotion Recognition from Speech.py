# Imports
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import warnings
warnings.filterwarnings('ignore')

# Path to dataset
DATA_PATH = "ravdess/"  # Change this to your dataset folder

# Emotion code mapping for RAVDESS
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Feature extraction using MFCC
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)  # normalize
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print("Error:", file_path, e)
        return None

# Load and label dataset
def load_data():
    X, y = [], []
    for file in os.listdir(DATA_PATH):
        if file.endswith('.wav'):
            emotion_code = file.split("-")[2]
            emotion = emotion_map.get(emotion_code)
            if emotion:
                features = extract_features(os.path.join(DATA_PATH, file))
                if features is not None:
                    X.append(features)
                    y.append(emotion)
    return np.array(X), np.array(y)

# Prepare data
X, y = load_data()
X = X[..., np.newaxis]  # Add channel dimension for CNN
le = LabelEncoder()
y_encoded = to_categorical(le.fit_transform(y))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create CNN model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(40, 174, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile and train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/emotion_model.h5")