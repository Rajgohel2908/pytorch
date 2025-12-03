import torch
import torch.nn as nn
import torch.optim as optim

sentences = ["i love this", "this is good", "so happy", "amazing work", 
             "i hate this", "this is bad", "so sad", "terrible work"]
labels = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

word2idx = {"<PAD>": 0, "i": 1, "love": 2, "this": 3, "is": 4, "good": 5, 
            "so": 6, "happy": 7, "amazing": 8, "work": 9, "hate": 10, 
            "bad": 11, "sad": 12, "terrible": 13}
vocab_size = len(word2idx)

def encode_sentence(sent):
    return torch.tensor([[word2idx.get(w,0) for w in sent.split()]])

class sentimentRNN(nn.Module):
    def __init__(self):
        super(sentimentRNN, self).__init__()
        self.embadding = nn.Embedding(vocab_size, 5)
        self.rnn = nn.LSTM(5, 10, batch_first=True)
        self.fc = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embadding(x)

        output, (hidden, cell) = self.rnn(x)

        final_memory = hidden.squeeze(0)

        out = self.fc(final_memory)
        return self.sigmoid(out)

model = sentimentRNN()
loss_fc = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    total_loss = 0
    for sent, label in zip(sentences, labels):
        optimizer.zero_grad()

        input = encode_sentence(sent)
        target = torch.tensor([[label]])

        pred = model(input)
        loss = loss_fc(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"epoch : [{epoch+1}] : loss : [{total_loss:.4f}]")

test_sent = "i hate you"
test_vec = encode_sentence(test_sent)
prediction = model(test_vec).item()

print(f"sentence : {test_sent}")
if prediction > 0.5:
    print("Positive")
else:
    print("Negative")