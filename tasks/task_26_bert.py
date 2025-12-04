from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="bert-base-multilingual-uncased-sentiment")

sentences = [
    "I love PyTorch, it's the best!",
    "This code is full of bugs, I hate it.",
    "The movie was okay, not great not bad.",
    "Transformers are mind-blowing!"
]

for sent in sentences:
    result = classifier(sent)

    label = result[0]['label']
    score = result[0]['score']

    print(f"sentence : '{sent}'")
    print(f"verdict : {label} and confidence : {score:4f}")

test_sent = "I love this!"
pred = classifier(test_sent)
label = pred[0]['label']
print(f"\n\ntest sentence : {test_sent}\npredicted : {label}")