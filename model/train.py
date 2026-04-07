import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data(file_path, label):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts, [label] * len(texts)

def main():
    up_texts, up_labels = load_data('data/left.txt', 1)       # класс 1 = влево
    down_texts, down_labels = load_data('data/right.txt', 2)  # класс 2 = вправо
    other_texts, other_labels = load_data('data/other.txt', 0)  # класс 0 = прочее

    all_texts = up_texts + down_texts + other_texts
    all_labels = up_labels + down_labels + other_labels
    print(f"Загружено примеров: left={len(up_texts)}, right={len(down_texts)}, other={len(other_texts)}")

    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, all_labels,
        test_size=0.2,
        random_state=42,
        stratify=all_labels
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=20000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    print("\nОценка модели")
    print(f"Точность: {accuracy_score(y_test, y_pred):.4f}")
    print("\nМатрица ошибок:")
    print(confusion_matrix(y_test, y_pred))
    print("\nИтоговые результаты:")
    print(classification_report(y_test, y_pred, target_names=['other', 'left', 'right']))

    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    joblib.dump(clf, 'models/classifier.pkl')

if __name__ == '__main__':
    main()