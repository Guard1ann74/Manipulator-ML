import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

def load_data(file_path, label):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts, [label] * len(texts)

def main():
    os.makedirs('models', exist_ok=True)

    forward_texts, forward_labels = load_data('data/forward.txt', 1)   # вперёд
    backward_texts, backward_labels = load_data('data/backward.txt', 2) # назад
    left_texts, left_labels = load_data('data/left.txt', 3)             # влево
    right_texts, right_labels = load_data('data/right.txt', 4)          # вправо
    other_texts, other_labels = load_data('data/other.txt', 0)          # прочее

    all_texts = forward_texts + backward_texts + left_texts + right_texts + other_texts
    all_labels = forward_labels + backward_labels + left_labels + right_labels + other_labels

    print(f"Загружено: forward={len(forward_texts)}, backward={len(backward_texts)}, left={len(left_texts)}, right={len(right_texts)}, other={len(other_texts)}")
    print(f"Всего примеров: {len(all_texts)}")

    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, all_labels,
        test_size=0.2,
        random_state=42,
        stratify=all_labels
    )
    print(f"Обучающая выборка: {len(X_train)},\nтестовая: {len(X_test)}")

    #Векторизация TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=15000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Обучение классификатора
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train_vec, y_train)

    # Предсказания и вероятности
    y_pred = clf.predict(X_test_vec)
    y_prob = clf.predict_proba(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f} ({acc:.2%})")

    ll = log_loss(y_test, y_prob)
    print(f"Log-loss: {ll:.4f}")

    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"Cohen's Kappa: {kappa:.4f}")

    target_names = ['other', 'forward', 'backward', 'left', 'right']
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    print("\n▶ Confusion matrix:")
    print(cm)

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='weighted')
    print(f"\nROC-AUC: {roc_auc:.4f}")

    # 9. Сохранение модели и данных
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    joblib.dump(clf, 'models/classifier.pkl')
    joblib.dump((X_test, y_test), 'models/test_data.pkl')

if __name__ == '__main__':
    main()