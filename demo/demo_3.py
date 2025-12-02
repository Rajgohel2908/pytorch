import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. DATA (Training Data) ---
# 1 = Positive, 0 = Negative
sentences = ["i love this", "this is good", "so happy", "amazing work", 
             "i hate this", "this is bad", "so sad", "terrible work"]
labels = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

# --- 2. VOCABULARY (Manually bana raha hoon simple rakhne ke liye) ---
# Real life mein hum tokenizer use karte hain
word2idx = {"<PAD>": 0, "i": 1, "love": 2, "this": 3, "is": 4, "good": 5, 
            "so": 6, "happy": 7, "amazing": 8, "work": 9, "hate": 10, 
            "bad": 11, "sad": 12, "terrible": 13}
vocab_size = len(word2idx) # 14 words

# Helper function: Sentence to Tensor
def encode_sentence(sent):
    return torch.tensor([[word2idx.get(w, 0) for w in sent.split()]])

# --- 3. THE MODEL (RNN Classifier) ---
class SentimentRNN(nn.Module):
    def __init__(self):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 5) # 5 dim vectors
        self.rnn = nn.RNN(5, 10, batch_first=True)   # 10 dim memory
        self.fc = nn.Linear(10, 1)                   # 10 -> 1 (Score)
        self.sigmoid = nn.Sigmoid()                  # 0-1 Probability

    def forward(self, x):
        # 1. Embed
        x = self.embedding(x)
        
        # 2. RNN Run karo
        # output shape: (batch, seq, hidden)
        # hidden shape: (1, batch, hidden)
        output, hidden = self.rnn(x)
        
        # 3. CRITICAL STEP: Sirf Aakhri Memory uthao!
        # Hum 'hidden' use kar sakte hain
        # hidden shape is (1, 1, 10) -> Squeeze karke (1, 10) banao
        final_memory = hidden.squeeze(0) 
        
        # 4. Decision
        out = self.fc(final_memory)
        return self.sigmoid(out)

# --- 4. TRAINING ---
model = SentimentRNN()
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("Training Sentiment AI... 🧠")
for epoch in range(100):
    total_loss = 0
    for sent, label in zip(sentences, labels):
        optimizer.zero_grad()
        
        # Input ready karo
        inputs = encode_sentence(sent) 
        target = torch.tensor([[label]]) # Shape (1, 1)
        
        # Forward -> Backward
        pred = model(inputs)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss {total_loss:.4f}")

# --- 5. TEST ---
test_sent = "i love work"
test_vec = encode_sentence(test_sent)
prediction = model(test_vec).item()

print(f"\nSentence: '{test_sent}'")
print(f"Positivity Score: {prediction:.4f}")
if prediction > 0.5:
    print("Result: Positive 😊")
else:
    print("Result: Negative 😡")