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


class SimpleLSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.Wf = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.01
        self.bf = np.zeros((hidden_dim,))
        self.Wi = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.01
        self.bi = np.zeros((hidden_dim,))
        self.Wc = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.01
        self.bc = np.zeros((hidden_dim,))
        self.Wo = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.01
        self.bo = np.zeros((hidden_dim,))

        self.Wy = np.random.randn(hidden_dim, output_dim) * 0.01
        self.by = np.zeros((output_dim,))

    def forward(self, x):
        h = np.zeros((len(x), self.hidden_dim))
        c = np.zeros((len(x), self.hidden_dim))
        y_pred = []

        for t in range(len(x)):
            xt = np.zeros((self.input_dim,))
            xt[x[t]] = 1
            combined = np.concatenate((xt, h[t - 1] if t > 0 else np.zeros_like(h[0])), axis=0)

            ft = self.sigmoid(np.dot(combined, self.Wf) + self.bf)
            it = self.sigmoid(np.dot(combined, self.Wi) + self.bi)
            ct_tilde = np.tanh(np.dot(combined, self.Wc) + self.bc)
            ct = ft * (c[t - 1] if t > 0 else np.zeros_like(c[0])) + it * ct_tilde
            ot = self.sigmoid(np.dot(combined, self.Wo) + self.bo)
            ht = ot * np.tanh(ct)

            h[t] = ht
            c[t] = ct
            yt = np.dot(ht, self.Wy) + self.by  #
            y_pred.append(yt)

        y_pred = np.array(y_pred)
        return y_pred, h, c

    def backward(self, x, y, y_pred, h, c, learning_rate):
        dWf = np.zeros_like(self.Wf)
        dbf = np.zeros_like(self.bf)
        dWi = np.zeros_like(self.Wi)
        dbi = np.zeros_like(self.bi)
        dWc = np.zeros_like(self.Wc)
        dbc = np.zeros_like(self.bc)
        dWo = np.zeros_like(self.Wo)
        dbo = np.zeros_like(self.bo)
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros((self.hidden_dim,))
        dc_next = np.zeros((self.hidden_dim,))

        for t in reversed(range(len(x))):
            xt = np.zeros((self.input_dim,))
            xt[x[t]] = 1  # One-hot encoding
            combined = np.concatenate((xt, h[t - 1] if t > 0 else np.zeros_like(h[0])), axis=0)

            dy = y_pred[t] - y
            dWy += np.outer(h[t], dy)
            dby += dy
            dh = np.dot(dy, self.Wy.T) + dh_next
            do = dh * np.tanh(c[t])
            dWo += np.outer(combined, self.sigmoid_derivative(do))
            dbo += self.sigmoid_derivative(do)

            dc = dh * h[t] * (1 - np.tanh(c[t]) ** 2) + dc_next
            dc_next = dc
            dWf += np.outer(combined, self.sigmoid_derivative(dc * c[t - 1]))
            dbf += self.sigmoid_derivative(dc * c[t - 1])
            dWi += np.outer(combined, self.sigmoid_derivative(dc * np.tanh(c[t])))
            dbi += self.sigmoid_derivative(dc * np.tanh(c[t]))

        # Update weights
        self.Wf -= learning_rate * dWf
        self.bf -= learning_rate * dbf
        self.Wi -= learning_rate * dWi
        self.bi -= learning_rate * dbi
        self.Wc -= learning_rate * dWc
        self.bc -= learning_rate * dbc
        self.Wo -= learning_rate * dWo
        self.bo -= learning_rate * dbo
        self.Wy -= learning_rate * dWy
        self.by -= learning_rate * dby

    def train(self, X, y, epochs=10, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0
            print(f"\nEpoch {epoch + 1}/{epochs}")
            for i in tqdm(range(len(X)), desc="Training Progress", leave=False):
                x_seq = X[i]
                y_true = y[i]
                y_pred, h, c = self.forward(x_seq)
                total_loss += np.sum((y_pred[-1] - y_true) ** 2) / len(y_true)
                self.backward(x_seq, y_true, y_pred, h, c, learning_rate)

            print(f"Loss: {total_loss:.4f}")

    def predict(self, X):
        y_pred = []
        for i in tqdm(range(len(X)), desc="Testing Progress"):
            x_seq = X[i]
            pred, _, _ = self.forward(x_seq)
            y_pred.append(pred[-1])
        return np.array(y_pred)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)


# Main Code
if __name__ == "__main__":
    train_df = pd.read_csv('./train_data.csv')
    test_df = pd.read_csv('./test_data.csv')

    tokenized_questions_train, tokenized_answers_train, y_train, vocab, label_keys = preprocess_text(train_df)
    tokenized_questions_test, tokenized_answers_test, y_test, _, _ = preprocess_text(test_df, vocab_size=len(vocab))

    X_train = [q + a for q, a in zip(tokenized_questions_train, tokenized_answers_train)]
    X_test = [q + a for q, a in zip(tokenized_questions_test, tokenized_answers_test)]

    input_dim = len(vocab)
    hidden_dim = 128
    output_dim = len(label_keys)
    lstm = SimpleLSTM(input_dim, hidden_dim, output_dim)

    print("Training...")
    lstm.train(X_train, y_train, epochs=5, learning_rate=0.01)

    print("\nEvaluating on test data...")
    y_pred = lstm.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    accuracy = np.mean(y_pred_binary == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    per_label_accuracy = np.mean(y_pred_binary == y_test, axis=0)
    for idx, label in enumerate(label_keys):
        print(f"Accuracy for label '{label}': {per_label_accuracy[idx]:.4f}")