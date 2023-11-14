# Dimensions Overview

**Queries (Q)**:

Dimension: `[batch_size, num_queries, depth_q]`
Each query is a vector of length depth_q. If you have multiple queries (like in the case of multiple words or tokens in a sequence), then num_queries represents the number of such queries.

**Keys (K) and Values (V)**:

Keys Dimension: `[batch_size, num_keys, depth_k]`
Values Dimension: `[batch_size, num_values, depth_v]`
Each key is a vector of length depth_k, and each value is a vector of length depth_v. Typically, num_keys and num_values are the same, representing the number of tokens in the input sequence.

## Example

Let's consider a simple example to illustrate these dimensions:

Assume we have a batch size of 1 (processing one sequence at a time).

Our input sequence consists of 4 tokens (words), so num_keys = num_values = 4.
We decide to have a depth of 3 for our queries, keys, and values for simplicity.

**Queries (Q):**

Say we are processing 2 queries. Then, num_queries = 2.
Dimension of Q: [1, 2, 3]

**Keys (K):**

Dimension of K: [1, 4, 3]

**Values (V):**

Dimension of V: [1, 4, 3]

## How They Interact

**Dot Product of Q and K:**

When we compute the dot product of Q and K, **we essentially measure the similarity between each query and each key**.
This operation results in a matrix of dimensions `[num_queries, num_keys]` for each batch. For our example, it would be `[2, 4]`.

**Applying Attention to V:**

The attention weights derived from the `QK` dot product (after applying softmax) are used to weight the values in V.

The final output would have dimensions `[1, num_queries, depth_v]`, which in our case is `[1, 2, 3]`.

In this way, the dimensions of Q, K, and V play a crucial role in how the transformer models process and understand the input sequence through the attention mechanism.

# How the Embeddings are used

## Initial Embeddings:

The process starts with word embeddings, which are vector representations of words. These embeddings capture the semantic properties of the words.
In a transformer model, each word in the input sequence is converted into an embedding vector using an embedding layer.

## Transformation into Q, K, V:

These embedding vectors are then transformed to create the Queries (Q), Keys (K), and Values (V) for the attention mechanism.
This transformation is usually a linear operation, like a matrix multiplication with a trainable weight matrix, followed by an optional bias addition. **In essence, for each embedding vector, you create three new vectors** – one each for Q, K, and V.

The purpose of this transformation is to prepare the embeddings for the attention mechanism, **allowing the model to focus on different aspects of the embeddings in different attention heads.**

## Use in Attention Mechanism:

The transformed vectors `(Q, K, V)` are then used in the attention mechanism.
The attention mechanism, specifically the scaled dot-product attention, calculates the dot product of the queries $(Q)$ with the keys $(K)$, scales these scores, applies a softmax to get the attention weights, and then uses these weights to create a weighted sum of the values $(V)$.

This process effectively allows the model to focus on different parts of the input sequence when producing each part of the output sequence, based on the learned importance (or attention) of each part.


## Role of Embeddings:

So, while the original embeddings themselves are not directly used in the calculation of attention scores, **they are the starting point and are crucial for generating the Q, K, and V vectors.**
The quality and information contained in these initial embeddings significantly impact the performance and capabilities of the transformer model.

