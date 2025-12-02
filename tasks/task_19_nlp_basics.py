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
    
    def decode(self, indices):
        decoded = []
        for idx in indices:
            word =self.idx2word.get(idx, "<UNK>")
            decoded.append(word)
        return " ".join(decoded)

data = [
    "I love PyTorch",
    "I love coding",
    "AI is the future",
    "Python is easy"
]

tokenize = tokenization()
tokenize.build_vocab(data)

test_sentence = "I love Python and AI"
print(f"Original : {test_sentence}")

encoded = tokenize.encode(test_sentence)
print(f"encoded : {encoded}")

decode = tokenize.decode(encoded)
print(f"decode : {decode}")