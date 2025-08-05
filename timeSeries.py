# timeSeries.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tcn import TCN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested

# CNN Model Definition
def make_cnn(input_shape, n_classes):
    return models.Sequential([
        layers.Conv1D(64, 5, activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])

# TCN Model Definition
def make_tcn(input_shape, n_classes):
    return models.Sequential([
        TCN(input_shape=input_shape, nb_filters=32, kernel_size=4, dilations=[1, 2, 4, 8], padding='causal', dropout_rate=0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])

def train_time_series_model(windows, model_name, epochs=40, batch_size=128, validation_split=0.1, random_seed=42):
    # Extract arrays
    X_array = np.stack(windows["signal"].values)
    y_labels = windows["activity"].astype(str).values
    groups = windows["userid"].values

    # Encode labels
    le = LabelEncoder()
    y_int = le.fit_transform(y_labels)
    n_classes = len(le.classes_)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y_int, test_size=0.3, stratify=groups, random_state=random_seed
    )

    if model_name == "cnn":
        model = make_cnn(input_shape=X_array.shape[1:], n_classes=n_classes)
    elif model_name == "tcn":
        model = make_tcn(input_shape=X_array.shape[1:], n_classes=n_classes)
    else:
        raise ValueError(f"Unsupported time-series model: {model_name}")

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=2
    )

    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print("\nTest accuracy:", test_acc)

    y_pred = model.predict(X_test).argmax(axis=1)
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)
    print("\nClassification report:\n")
    print(classification_report(y_test_labels, y_pred_labels))


def train_kmeans_timeseries(windows, n_clusters=6, method="sklearn"):
    # Extract signal and labels
    X_array = np.stack(windows["signal"].values)  # Shape: (N, 512, 3)
    y_labels = windows["activity"].astype(str).values

    print(f"Data shape: {X_array.shape}")

    if method == "sklearn":
        # Flatten each signal (512, 3) → (1536,)
        X_flat = X_array.reshape((X_array.shape[0], -1))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        y_pred = kmeans.fit_predict(X_flat)

    elif method == "sktime":
        # Convert to sktime format: 2D nested pandas DataFrame
        X_nested = from_3d_numpy_to_nested(X_array)

        kmeans = TimeSeriesKMeans(n_clusters=n_clusters, random_state=42)
        y_pred = kmeans.fit_predict(X_nested)

    else:
        raise ValueError("Unsupported method. Choose 'sklearn' or 'sktime'.")

    # Encode true labels
    le = LabelEncoder()
    y_true = le.fit_transform(y_labels)

    print("\nContingency table of true vs predicted clusters:")
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_true, y_pred))

    # Optionally: Silhouette Score (sklearn-based only)
    if method == "sklearn":
        sil = silhouette_score(X_flat, y_pred)
        print(f"\nSilhouette Score: {sil:.3f}")