import datasets

dataset = datasets.load_dataset('tiny_shakespeare')

# Train, test, val split 
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Train, test, val split text 
train_text = train_data[0]['text']
val_text = val_data[0]['text']
test_text = test_data[0]['text']

# Dataset params
chars = sorted(list(set(train_text)))
vocab_size = len(chars)

