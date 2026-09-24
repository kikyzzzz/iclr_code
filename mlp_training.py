import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "dataset_path"

def load_saved_dataset(file_path):
    data = np.load(file_path, allow_pickle=True)
    dataset = {
        "X_train": data["X_train"],
        "X_test": data["X_test"],
        "y_train": data["y_train"],
        "y_test": data["y_test"],
        "Z_train": data["Z_train"],
        "Z_test": data["Z_test"],
        "feature_names": data["feature_names"].tolist(),
        "metadata": dict(data["metadata"].item()) if "metadata" in data else {}
    }
    return dataset

def train_mlp_classifier(dataset, hidden_layer_sizes=(128, 64), max_iter=1000, activation='relu', solver='adam', random_state=42):
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        verbose=False,
        n_iter_no_change=10,
        tol=0.005,
        alpha=0.001,
    )

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    y_pred_proba = mlp.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    target_fpr = 0.05
    if fpr[-1] < target_fpr:
        tpr_at_5fpr = tpr[-1]
    elif fpr[0] > target_fpr:
        tpr_at_5fpr = tpr[0]
    else:
        idx = np.where(fpr >= target_fpr)[0][0]
        tpr_at_5fpr = tpr[idx - 1] + (target_fpr - fpr[idx - 1]) * (tpr[idx] - tpr[idx - 1]) / (fpr[idx] - fpr[idx - 1])

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'tpr_at_5fpr': tpr_at_5fpr,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred)
    }

    return mlp, metrics

if __name__ == "__main__":
    dataset = load_saved_dataset(DATASET_PATH)
    model, metrics = train_mlp_classifier(dataset)