# Explanation of the transformer architecture

## High level overview

**Linear Algebra and Embeddings**: Transformers start by converting input data (like text) into vectors using embeddings. These vectors are representations in a high-dimensional space, where each dimension represents a feature of the input data. This is grounded in linear algebra, specifically in the use of matrices and vectors.

**Positional Encoding**: Since transformers do not inherently process sequential data in order (like previous architectures, e.g., RNNs or LSTMs), they require positional encodings to retain the order of the sequence. This is usually done by adding a vector to each input embedding, which represents the position of each element in the sequence.

**Attention Mechanism**: The core of a transformer is its attention mechanism. The attention mechanism allows the model to focus on different parts of the input sequence when producing each part of the output sequence. This is akin to how humans pay attention to different words in a sentence when trying to understand its meaning. Mathematically, this involves calculating scores (using dot products) for each pair of input and output positions, applying a softmax function to these scores to obtain probabilities, and then using these probabilities to weight the input representations.

**Feedforward Neural Networks**: Each layer of a transformer contains a feedforward neural network, which applies a series of linear (matrix multiplication plus bias addition) and non-linear (like ReLU) operations to the transformed inputs. These networks are responsible for the actual 'learning' in the model.

**Layer Normalization and Residual Connections**: Transformers also use techniques like layer normalization and residual connections to stabilize and accelerate training. Layer normalization involves normalizing the outputs of each layer to have a mean of zero and a standard deviation of one, while residual connections involve adding the input of each layer to its output, which helps mitigate the vanishing gradient problem.

**Probability and Loss Functions**: Finally, in the context of training, transformers, like other neural networks, use probability theory. They typically output probabilities (using functions like softmax) for each class or token they are predicting, and they are trained using loss functions like cross-entropy, which measure the difference between the predicted probabilities and the actual labels.

## More details

We go over each topic of the high level overview in each of the subfolder.