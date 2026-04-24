import torch
import pickle

EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_CLASSES = 5
MAX_LEN = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model_2.train_nn import CommandClassifier

# Загрузка словаря
with open('model_2/models/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
VOCAB_SIZE = len(vocab)

# Загрузка модели
model = CommandClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('model_2/models/nn_model_best.pth', map_location=DEVICE))
model.eval()


def text_to_sequence(text, max_len=MAX_LEN):
    words = text.lower().split()[:max_len]
    seq = [vocab.get(word, 1) for word in words]
    seq += [0] * (max_len - len(seq))
    return seq


def recognize(text, threshold=0.6):
    seq = text_to_sequence(text)
    input_tensor = torch.LongTensor([seq]).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    label = pred.item()
    conf = confidence.item()

    if conf < threshold:
        return -1, conf
    return label, conf


label_to_command = {
    0: None,
    1: "переместить объект вперёд",
    2: "переместить объект назад",
    3: "переместить объект влево",
    4: "переместить объект вправо"
}


def check_answer(user_input):
    label, conf = recognize(user_input, threshold=0.6)

    if label == -1 or label == 0:
        return 0, f"Действие не распознано | Уверенность: {conf:.2%}", -1
    else:
        return 1, f"Действие выполнено: {label_to_command[label]} | Уверенность: {conf:.2%}", label