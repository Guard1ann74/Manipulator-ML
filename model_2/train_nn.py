import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score,
    accuracy_score
)
from sklearn.preprocessing import label_binarize
import os
import pickle
import numpy as np

EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_CLASSES = 5
MAX_LEN = 10
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CommandClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

def load_data(file_path, label):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip().lower() for line in f if line.strip()]
    return texts, [label] * len(texts)


def main():
    os.makedirs('models', exist_ok=True)

    forward_texts, forward_labels = load_data('data/forward.txt', 1)
    backward_texts, backward_labels = load_data('data/backward.txt', 2)
    left_texts, left_labels = load_data('data/left.txt', 3)
    right_texts, right_labels = load_data('data/right.txt', 4)
    other_texts, other_labels = load_data('data/other.txt', 0)

    all_texts = forward_texts + backward_texts + left_texts + right_texts + other_texts
    all_labels = forward_labels + backward_labels + left_labels + right_labels + other_labels

    print(f"Загружено: forward={len(forward_texts)}, "
          f"backward={len(backward_texts)}, "
          f"left={len(left_texts)}, "
          f"right={len(right_texts)}, "
          f"other={len(other_texts)}")
    print(f"Всего примеров: {len(all_texts)}")

    def build_vocab(texts, min_freq=1):
        word_counts = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1
        }
        for word, count in word_counts.items():
            if count >= min_freq:
                vocab[word] = len(vocab)
        return vocab

    vocab = build_vocab(all_texts)
    VOCAB_SIZE = len(vocab)
    print(f"Размер словаря: {VOCAB_SIZE}")

    with open('models/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    def text_to_sequence(text, max_len=MAX_LEN):
        words = text.split()[:max_len]
        seq = [vocab.get(word, 1) for word in words]
        seq += [0] * (max_len - len(seq))
        return seq

    X = [text_to_sequence(text) for text in all_texts]
    y = all_labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    class CommandDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.LongTensor(X)
            self.y = torch.LongTensor(y)
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = CommandDataset(X_train, y_train)
    test_dataset = CommandDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CommandClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters())}")

    # Обучение
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nНачало обучения")
    print("-" * 60)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>8} | {'Val Kappa':>10}")
    print("-" * 60)

    best_val_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(y_batch.cpu().numpy())

        val_loss /= len(test_loader)
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_kappa = cohen_kappa_score(all_val_labels, all_val_preds)

        print(f"{epoch+1:6d} | {total_loss/len(train_loader):10.4f} | {val_loss:10.4f} | {val_acc:8.4f} | {val_kappa:10.4f}")

        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/nn_model_best.pth')

    print("-" * 60)

    # Оценка на тесте
    model.eval()
    all_preds = []
    all_labels_list = []
    all_probs = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels_list.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)

    print("Результаты")

    # Accuracy
    final_acc = accuracy_score(all_labels_list, all_preds)
    print(f"\nAccuracy: {final_acc:.4f} ({final_acc:.2%})")

    # Cohen's Kappa
    kappa = cohen_kappa_score(all_labels_list, all_preds)
    print(f"Cohen's Kappa: {kappa:.4f}")

    # ROC-AUC (One-vs-Rest)
    y_test_bin = label_binarize(all_labels_list, classes=[0, 1, 2, 3, 4])

    roc_auc = roc_auc_score(y_test_bin, all_probs, multi_class='ovr', average='weighted')
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Classification report
    print("\nClassification report:")
    target_names = ['other', 'forward', 'backward', 'left', 'right']
    print(classification_report(all_labels_list, all_preds, target_names=target_names))

    # Confusion matrix
    print("Confusion matrix:")
    cm = confusion_matrix(all_labels_list, all_preds)
    print(cm)

    total_errors = len(all_labels_list) - (cm.trace())
    print(f"\nОбщее количество ошибок: {total_errors} из {len(all_labels_list)} ({total_errors/len(all_labels_list):.2%})")

    # Сохраняем финальную модель
    torch.save(model.state_dict(), 'models/nn_model.pth')
    print("\nМодель сохранена в models/nn_model.pth")
    print(f"Лучшая модель (best val acc={best_val_acc:.4f}) сохранена в models/nn_model_best.pth")


if __name__ == '__main__':
    main()