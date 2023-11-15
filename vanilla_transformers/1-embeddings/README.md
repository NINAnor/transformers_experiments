## Embeddings
In a transformer, the input data (like words in a sentence) need to be represented in a way that a neural network can process. This is where embeddings come in.

### How Embeddings Work:

- **Vocabulary**: Assume we have a fixed vocabulary of size `V`. Each word (or token) in this vocabulary is represented by a unique one-hot encoded vector of size `V`. In one-hot encoding, the vector is all zeros except for a single '1' at the index representing the word.
- **Embedding Matrix**: An embedding matrix `E` of size `V x d` is created, where `d` is the chosen dimensionality of the embeddings. Each row of `E` represents the embedding of a word in `d`-dimensional space.
- **Word to Vector**: To find the embedding of a word, we multiply its one-hot vector by the embedding matrix `E`. Mathematically, for a word with one-hot vector `w`, its embedding is `wE`.

#### Example:
- Suppose our vocabulary has 10,000 words and we choose an embedding size of 512.
- The embedding matrix `E` would be `10000 x 512`.
- For a word represented by a one-hot vector `w` (where `w` is a `10000`-dimensional vector with one '1' and 9999 '0's), its embedding is the corresponding row in `E`.

## Positional Encoding
Transformers do not have a recurrent structure and hence do not inherently understand sequence order. Positional encodings add information about the position of each word in the sequence to the embeddings.

### How Positional Encodings Work:

- **Equations**: The positional encodings use sine and cosine functions of different frequencies. For position `pos` and dimension `i`, the positional encoding `PE(pos, i)` is calculated as:
  - `PE(pos, 2i) = sin(pos / 10000^(2i/d))`
  - `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`
- Here, `pos` is the position in the sequence, and `i` is the dimension. This pattern ensures that each position gets a unique encoding but maintains consistency across dimensions.

#### Example:
- If our sequence length is 50 and the embedding size is 512, each word in the sequence will be assigned a positional encoding vector of the same size (512).
- For the first word (position 0), `PE(0, i)` is calculated for each dimension `i` in the embedding using the sine and cosine equations.
- This positional encoding vector is then added to the embedding vector of the word.

## Combined Word and Positional Embeddings
In practice, the transformer combines these two types of embeddings by element-wise addition of the word embedding and its corresponding positional encoding. This sum gives the transformer model information about both the identity of the words and their positions in the sequence.

### Importance in Transformers
This combination of word embeddings and positional encodings is crucial. It allows the model to understand not only which words are in the input but also their order. This is essential for tasks like language translation or text generation, where the meaning depends heavily on word order.