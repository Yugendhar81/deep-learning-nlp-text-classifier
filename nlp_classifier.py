
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

# Sample text dataset
texts = [
    "I love this product",
    "This is amazing",
    "I am very happy",
    "Excellent service",
    "I hate this product",
    "This is terrible",
    "I am very disappointed",
    "Worst experience ever"
]

labels = [1, 1, 1, 1, 0, 0, 0, 0]

# Text preprocessing
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# Build vocabulary
counter = Counter()
for text in texts:
    counter.update(tokenize(text))

vocab = {"<PAD>": 0, "<UNK>": 1}
for word in counter:
    vocab[word] = len(vocab)

# Convert text into numerical sequences
def encode(text, max_len=6):
    ids = [vocab.get(word, 1) for word in tokenize(text)]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids

X = torch.tensor([encode(t) for t in texts], dtype=torch.long)
y = torch.tensor(labels, dtype=torch.float32)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Neural network text classifier
class TextClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16, padding_idx=0)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        mask = (x != 0).unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp(min=1)
        return self.fc(pooled).squeeze(1)

model = TextClassifier(len(vocab))
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(50):
    model.train()
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Predict sentiment
def predict_sentiment(text):
    model.eval()
    with torch.no_grad():
        x = torch.tensor([encode(text)], dtype=torch.long)
        probability = torch.sigmoid(model(x)).item()
    label = "Positive" if probability >= 0.5 else "Negative"
    return label, probability

print("\nSample prediction:")
print(predict_sentiment("I love this amazing product"))
print(predict_sentiment("This is a terrible experience"))
