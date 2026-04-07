import joblib

vectorizer = joblib.load('models/vectorizer.pkl')
clf = joblib.load('models/classifier.pkl')

label_to_command = {
    0: None,
    1: "переместить объект влево",
    2: "переместить объект вправо"
}


def recognize(text, threshold=0.6):
    X = vectorizer.transform([text])
    pred_label = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    confidence = proba[pred_label]

    if confidence < threshold:
        return -1, confidence
    return pred_label, confidence


def main():
    print("Система распознавания команд манипулятора")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ('выход', 'exit', 'quit'):
            break
        if not user_input:
            continue
        label, conf = recognize(user_input, threshold=0.6)
        if label == -1 or label == 0:
            print(f"Действие не распознано | Точность: {conf:.2%}")
        else:
            print(f"Действие выполнено: {label_to_command[label]} | Точность: {conf:.2%}")

if __name__ == '__main__':
    main()