import torch
import torch.optim as optim
import torch.nn as nn

data = "Deep learning is amazing. Deep learning is the future. I love deep learning."
sentences = data.replace(".", "").split(". ")

class tokenization:
    def __init__(self):
        self.word2idx = {'<PAD>':0, '<UNK>':1}
        self.idx2word = {0:'<PAD>', 1:'<UNK>'}
        self.count = 2

    def build_vocab(self, sentence):
        for sent in sentence:
            words = sent.lower().split()

            for word in words:
                if word not in self.word2idx:
                    self.word2idx[word] = self.count
                    self.idx2word[self.count] = word
                    self.count += 1
        print(f"total words learned: {self.count}")

    def encode(self, sentence):
        words = sentence.lower().split()
        encoded = []
        for word in words:
            idx = self.word2idx.get(word, 1)
            encoded.append(idx)
        return encoded

tokenizer = tokenization()
tokenizer.build_vocab(sentences)

data_encode = tokenizer.encode(data.replace(".", ""))
inputs = []
targets = []

seq_len = 3

for i in range(len(data_encode) - seq_len):
    seq_in = data_encode[i: i + seq_len]
    target_out = data_encode[i + seq_len]

    inputs.append(seq_in)
    targets.append(target_out)

inputs = torch.tensor(inputs)
targets = torch.tensor(targets)

class textgenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(textgenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)

        out ,_ =self.lstm(x)
        last_out = out[:, -1, :]

        prediction = self.fc(last_out)

        return prediction

vocab_size = tokenizer.count
embed_dim = 10
hidden_dim = 20

model = textgenerator(vocab_size,embed_dim,hidden_dim)

loss_fc = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()

    output = model(inputs)
    loss = loss_fc(output, targets)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"epoch : {epoch + 1} , loss : {loss.item():.4f}")

test_sen = "deep learning is"
encoded_sen = torch.tensor([tokenizer.encode(test_sen)])

with torch.no_grad():
    prediction = model(encoded_sen)

    predicted_idx = torch.argmax(prediction, dim=1).item()

    predicted_word = tokenizer.idx2word[predicted_idx]
    print(f"AI Predicted Next Word: '{predicted_word}'")