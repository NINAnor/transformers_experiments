import torch
import torch.nn as nn
import pandas as pd
import spacy
from torch.utils.data import Dataset, DataLoader

# Load spacy model for tokenization
nlp = spacy.load("en_core_web_sm")

# Tokenization function using spacy
def tokenize(text):
    return [token.text.lower() for token in nlp.tokenizer(text)]

# Create a simple dataset
data = {
    'Sentence': ['Hello, how are you?', 'I am learning about AI!', 'Transformers are interesting.']
}
df = pd.DataFrame(data)
df.to_csv("sentences.csv", index=False)

# Manually create vocabulary
def build_vocab(dataframe):
    vocab = {}
    for sentence in dataframe['Sentence']:
        tokens = tokenize(sentence)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

vocab = build_vocab(df)
vocab_size = len(vocab)
embedding_dim = 64

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.dataframe = dataframe
        self.vocab = vocab

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['Sentence']
        return torch.tensor([self.vocab[token] for token in tokenize(text)], dtype=torch.long)

# Create dataset and dataloader
dataset = CustomDataset(df, vocab)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Embedding Layer
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length):
        super(TransformerEmbedding, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(max_length, embedding_dim)

    def forward(self, x):
        positions = torch.arange(x.size(0)).unsqueeze(0)
        return self.token_embeddings(x) + self.position_embeddings(positions)

# Initialize model
max_length = max([len(tokenize(sentence)) for sentence in df['Sentence']])
model = TransformerEmbedding(vocab_size, embedding_dim, max_length)

# Example input and embedding extraction
for batch in dataloader:
    embeddings = model(batch[0])
    print(embeddings)
    break  # Example for one batch, remove this to process all data