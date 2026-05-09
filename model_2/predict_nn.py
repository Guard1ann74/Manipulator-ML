import joblib
import numpy as np
import warnings
from sentence_transformers import SentenceTransformer
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
embedder = SentenceTransformer('cointegrated/rubert-tiny2')
clf = joblib.load('model_transform/intent_classifier.pkl')

label_to_command = {
    0: None,
    1: "переместить объект вперёд",
    2: "переместить объект назад",
    3: "переместить объект влево",
    4: "переместить объект вправо"
}


def recognize(text, threshold=0.75):
    text = text.lower().strip()
    emb = embedder.encode([text])
    probs = clf.predict_proba(emb)[0]
    max_prob = np.max(probs)
    label = clf.classes_[np.argmax(probs)]
    if max_prob < threshold:
        return -1, max_prob

    return label, max_prob


def check_answer(user_input):
    label, conf = recognize(user_input, threshold=0.45)
    emb = embedder.encode([user_input])
    raw_probs = clf.predict_proba(emb)[0]
    raw_label = clf.classes_[np.argmax(raw_probs)]
    raw_command = label_to_command.get(raw_label, "Other")

    if label == -1 or label == 0:
        return 0, f"Действие не распознано (Склонялся к: {raw_command}) | Уверенность: {conf:.2%}", -1
    else:
        return 1, f"Действие выполнено: {label_to_command[label]} | Уверенность: {conf:.2%}", label

if __name__ == "__main__":
    while True:
        text = input("> ")
        if text.lower() == 'exit':
            break
        status, msg, cmd = check_answer(text)
        print(msg)