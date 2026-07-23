from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

def evaluate_model(y_test, y_pred):

    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    print("\n========== MODEL EVALUATION ==========")

    print("\nAccuracy:")
    print(accuracy)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nPrecision:")
    print(precision)

    print("\nRecall:")
    print(recall)

    print("\nF1 Score:")
    print(f1)