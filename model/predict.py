import joblib

vectorizer = joblib.load('models/vectorizer.pkl')
clf = joblib.load('models/classifier.pkl')

label_to_command = {
    0: None,
    1: "подняться вверх",
    2: "опуститься вниз"
}

def recognize(text):
    X = vectorizer.transform([text])
    pred_label = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    confidence = proba[pred_label]
    return pred_label, confidence

def main():
    print("Система распознавания команд манипулятора")
    print("Доступные команды: 'вверх', 'вниз'")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ('выход', 'exit', 'quit'):
            break
        if not user_input:
            continue
        label, conf = recognize(user_input)
        if label == 0:
            print(f"Действие не распознано | Точность: {conf:.2%}")
        else:
            print(f"Действие выполнено: {label_to_command[label]} | Точность: {conf:.2%}")

if __name__ == '__main__':
    main()