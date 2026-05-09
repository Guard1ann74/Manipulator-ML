import os
import joblib
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score


def load_data(file_path, label):
    if not os.path.exists(file_path):
        print(f"Внимание: Файл {file_path} не найден!")
        return [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip().lower() for line in f if line.strip()]
    return texts, [label] * len(texts)


def main():
    os.makedirs('model_transform', exist_ok=True)

    forward_texts, forward_labels = load_data('data/forward.txt', 1)
    backward_texts, backward_labels = load_data('data/backward.txt', 2)
    left_texts, left_labels = load_data('data/left.txt', 3)
    right_texts, right_labels = load_data('data/right.txt', 4)
    other_texts, other_labels = load_data('data/other.txt', 0)

    all_texts = forward_texts + backward_texts + left_texts + right_texts + other_texts
    all_labels = forward_labels + backward_labels + left_labels + right_labels + other_labels
    embedder = SentenceTransformer('cointegrated/rubert-tiny2')

    X_all = embedder.encode(all_texts, show_progress_bar=True)
    all_labels_np = np.array(all_labels)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_clf = SVC(C=100.0, kernel='rbf', class_weight='balanced')
    cv_scores = cross_val_score(cv_clf, X_all, all_labels_np, cv=skf)
    print(f"Средняя точность на CV: {cv_scores.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, all_labels_np, test_size=0.2, random_state=42, stratify=all_labels_np
    )

    clf = SVC(C=100.0, kernel='rbf', probability=True, class_weight='balanced')
    clf.fit(X_train, y_train)

    test_preds = clf.predict(X_test)
    report = classification_report(y_test, test_preds,
                                   target_names=['Other', 'Forward', 'Backward', 'Left', 'Right'],
                                   output_dict=True)
    cm = confusion_matrix(y_test, test_preds)
    stats = {
        "cv_scores": cv_scores.tolist(),
        "cv_mean": cv_scores.mean(),
        "test_accuracy": accuracy_score(y_test, test_preds),
        "confusion_matrix": cm.tolist(),
        "class_names": ['Other', 'Forward', 'Backward', 'Left', 'Right'],
        "report": report,
        "embeddings": X_test.tolist(),
        "true_labels": y_test.tolist()
    }

    with open('model_transform/training_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    clf.fit(X_all, all_labels_np)
    joblib.dump(clf, 'model_transform/intent_classifier.pkl')
    print("Модель переобучена на полном датасете")


if __name__ == '__main__':
    main()