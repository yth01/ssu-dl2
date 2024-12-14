import numpy as np
import pandas as pd
from tqdm import tqdm


def preprocess_text(df, vocab_size=5000):
    from collections import Counter
    from itertools import chain

    questions = [q.lower().split() for q in df['question']]
    answers = [a.lower().split() for a in df['answer']]

    all_words = list(chain(*questions, *answers))
    word_counts = Counter(all_words)
    most_common_words = [word for word, _ in word_counts.most_common(vocab_size - 1)]
    vocab = {word: i + 1 for i, word in enumerate(most_common_words)}
    vocab['<UNK>'] = 0

    def tokenize(sentences):
        return [[vocab.get(word, 0) for word in sentence] for sentence in sentences]

    tokenized_questions = tokenize(questions)
    tokenized_answers = tokenize(answers)

    labels = df['label'].apply(lambda x: eval(x)).values
    label_keys = list(labels[0].keys())
    y = np.array([[int(label[key]) for key in label_keys] for label in labels])

    return tokenized_questions, tokenized_answers, y, vocab, label_keys


class SimpleRNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.Wx = np.random.randn(input_dim, hidden_dim) * 0.01
        self.Wh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.bh = np.zeros((hidden_dim,))
        self.Wo = np.random.randn(hidden_dim, output_dim) * 0.01
        self.bo = np.zeros((output_dim,))

    def forward(self, x):
        h = np.zeros((len(x), self.hidden_dim))
        y_pred = []

        for t in range(len(x)):
            xt = np.zeros((self.input_dim,))
            xt[x[t]] = 1
            h[t] = np.tanh(np.dot(xt, self.Wx) + np.dot(h[t - 1], self.Wh) + self.bh)
            yt = np.dot(h[t], self.Wo) + self.bo
            y_pred.append(yt)

        y_pred = np.array(y_pred)
        return y_pred, h

    def backward(self, x, y, y_pred, h, learning_rate):
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dbh = np.zeros_like(self.bh)
        dWo = np.zeros_like(self.Wo)
        dbo = np.zeros_like(self.bo)
        dh_next = np.zeros((self.hidden_dim,))

        for t in reversed(range(len(x))):
            xt = np.zeros((self.input_dim,))
            xt[x[t]] = 1
            dy = y_pred[t] - y
            dWo += np.outer(h[t], dy)
            dbo += dy
            dh = np.dot(dy, self.Wo.T) + dh_next
            dh_raw = (1 - h[t] ** 2) * dh
            dWx += np.outer(xt, dh_raw)
            dWh += np.outer(h[t - 1], dh_raw)
            dbh += dh_raw
            dh_next = np.dot(dh_raw, self.Wh.T)

        self.Wx -= learning_rate * dWx
        self.Wh -= learning_rate * dWh
        self.bh -= learning_rate * dbh
        self.Wo -= learning_rate * dWo
        self.bo -= learning_rate * dbo

    def train(self, X, y, epochs=10, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0
            print(f"\nEpoch {epoch + 1}/{epochs}")
            for i in tqdm(range(len(X)), desc="Training Progress", leave=False):
                x_seq = X[i]
                y_true = y[i]
                y_pred, h = self.forward(x_seq)
                total_loss += np.sum((y_pred[-1] - y_true) ** 2) / len(y_true)
                self.backward(x_seq, y_true, y_pred, h, learning_rate)

            print(f"Loss: {total_loss:.4f}")

    def predict(self, X):
        y_pred = []
        for i in tqdm(range(len(X)), desc="Testing Progress"):
            x_seq = X[i]
            pred, _ = self.forward(x_seq)
            y_pred.append(pred[-1])
        return np.array(y_pred)


# Main Code
if __name__ == "__main__":
    train_df = pd.read_csv('/content/train_data.csv')
    test_df = pd.read_csv('/content/test_data.csv')

    tokenized_questions_train, tokenized_answers_train, y_train, vocab, label_keys = preprocess_text(train_df)
    tokenized_questions_test, tokenized_answers_test, y_test, _, _ = preprocess_text(test_df, vocab_size=len(vocab))

    X_train = [q + a for q, a in zip(tokenized_questions_train, tokenized_answers_train)]
    X_test = [q + a for q, a in zip(tokenized_questions_test, tokenized_answers_test)]

    input_dim = len(vocab)
    hidden_dim = 128
    output_dim = len(label_keys)
    rnn = SimpleRNN(input_dim, hidden_dim, output_dim)

    print("Training...")
    rnn.train(X_train, y_train, epochs=5, learning_rate=0.01)

    print("\nEvaluating on test data...")
    y_pred = rnn.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    accuracy = np.mean(y_pred_binary == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    per_label_accuracy = np.mean(y_pred_binary == y_test, axis=0)
    for idx, label in enumerate(label_keys):
        print(f"Accuracy for label '{label}': {per_label_accuracy[idx]:.4f}")