# Normalization in Neural Networks and Transformers

Normalization is a technique of scaling the values of activations so that convergence happens more smoothly. When we backpropagate and perform gradient descent, normalization can make training smoother.

- There are several different types of normalization techniques, and they differ based on **which dimension we are normalizing on**.
- Traditionally in normalization techniques, we use $\gamma$ and $\beta$ as the learnable parameters. $\gamma$ is the scaling parameter and $\beta$ is the shifting parameter.

## Batch Norm

- Traditional CNNs used BatchNorm, which basically uses the other values in a batch to normalize a particular value.
- Mathematically: $\hat{x} = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}$
- Then: $y_i = \gamma_i(\hat{x_i}) + \beta_i$

For an example of a fully connected NN:

```text
Batch × Features
4 × 3

        f1   f2   f3
x1      2    5    8    ↓
x2      4    7    6    ↓
x3      6    9    4    ↓
x4      8    3    2    ↓
```

- BatchNorm normally calculates statistics for each feature across the batch.

### Why BatchNorm Isn't Ideal for Transformers

Transformers deal with sequences whose lengths can vary.

Suppose we have these three sentences:

```text
Sentence 1: "I love machine learning a lot"
Sentence 2: "I like cats"
Sentence 3: "I am a human"
```

The sentences have different lengths:

```text
Sentence 1 → 6 tokens
Sentence 2 → 3 tokens
Sentence 3 → 4 tokens
```

When we process them together in a batch, we usually pad the shorter sequences to the same `seq_len`:

```text
Sentence 1: "I love machine learning a lot"
Sentence 2: "I like cats <PAD> <PAD> <PAD>"
Sentence 3: "I am a human <PAD> <PAD>"
```

This gives the batch a fixed shape:

```text
(batch, seq_len, d_model)
```

BatchNorm calculates statistics using values across the batch/normalized dimension. With sequences, padding and variable lengths can therefore affect the statistics, making BatchNorm a less natural fit for Transformer representations.

This is one reason LayerNorm is commonly used instead.

## Layer Norm

- Was used by transformers until something better appeared.
- The fundamental difference is **BatchNorm is ↓ and LayerNorm is →**.
- For a tensor of shape `(batch, seq_len, d_model)`, LayerNorm computes mean and variance across the `d_model` numbers of each individual token.
- It does not normalize across the sentence. Each token is handled independently.
- The formula is exactly the same as BatchNorm. Only the direction along which we normalize differs.

### LayerNorm Example

For this example, let `d_model = 6`:

```text
Sentence 1: "I love machine learning"

I          → [0.2,  1.3, -0.5,  0.7,  0.2, -0.1]
love       → [0.7,  0.2,  1.1,  0.4,  0.8,  0.3]
machine    → [0.1,  0.9,  0.4, -0.2,  0.6,  1.0]
learning   → [0.5,  0.3,  0.8,  0.2, -0.1,  0.7]
```

LayerNorm works on one token at a time. For example:

```text
I → [0.2, 1.3, -0.5, 0.7, 0.2, -0.1]
     └─────────────────────────────────┘
              6 d_model values
```

It calculates the mean and variance across these 6 values, normalizes them, and then moves to the next token.

```text
I        → normalize its 6 values
love     → normalize its 6 values
machine  → normalize its 6 values
learning → normalize its 6 values
```

It does **not** take all the tokens in the sentence and calculate one mean and variance.

### LayerNorm With Padding

Now consider the three sentences from the BatchNorm example:

```text
Sentence 1: "I love machine learning a lot"
Sentence 2: "I like cats"
Sentence 3: "I am a human"
```

After padding to the longest sequence:

```text
Sentence 1: "I love machine learning a lot"

I          → [0.2,  1.3, -0.5,  0.7,  0.2, -0.1]
love       → [0.7,  0.2,  1.1,  0.4,  0.8,  0.3]
machine    → [0.1,  0.9,  0.4, -0.2,  0.6,  1.0]
learning   → [0.5,  0.3,  0.8,  0.2, -0.1,  0.7]
a          → [0.4,  0.6, -0.2,  0.9,  0.1,  0.5]
lot        → [0.8, -0.1,  0.3,  0.5,  0.7,  0.2]

Sentence 2: "I like cats <PAD> <PAD> <PAD>"

I          → [0.7,  0.2,  1.1,  0.4,  0.8,  0.3]
like       → [0.3,  0.9,  0.1,  0.6, -0.2,  0.5]
cats       → [0.6,  0.4,  0.8, -0.1,  0.3,  0.7]
<PAD>      → [0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
<PAD>      → [0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
<PAD>      → [0.0,  0.0,  0.0,  0.0,  0.0,  0.0]

Sentence 3: "I am a human <PAD> <PAD>"

I          → [0.9,  0.4,  0.3,  0.4,  0.6,  0.1]
am         → [0.2,  0.8,  0.5, -0.1,  0.7,  0.3]
a          → [0.5,  0.3,  0.9,  0.2, -0.2,  0.6]
human      → [0.7,  0.1,  0.4,  0.8,  0.5,  0.2]
<PAD>      → [0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
<PAD>      → [0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
```

Now the batch has the shape:

```text
(batch, seq_len, d_model)
(3,      6,       6)
```

The important point is that LayerNorm normalizes each token independently across its `d_model` dimensions. The `<PAD>` tokens therefore do not affect the LayerNorm statistics of real tokens such as `I`, `love`, or `cats`.

Padding is still important for the Transformer as a whole because attention needs to know which positions are padding. LayerNorm itself, however, operates independently on each token's `d_model`-dimensional representation.

### Full LayerNorm Calculation

For the token `I`:

```text
I → [0.2, 1.3, -0.5, 0.7, 0.2, -0.1]
```

Mean:

$$
\mu = \frac{0.2+1.3-0.5+0.7+0.2-0.1}{6} = 0.3
$$

Variance:

$$
\sigma^2 = \frac{(-0.1)^2+(1.0)^2+(-0.8)^2+(0.4)^2+(-0.1)^2+(-0.4)^2}{6} = 0.33
$$

Normalize:

$$
\hat{x_i}=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

Approximately:

```text
I → [-0.174, 1.740, -1.392, 0.696, -0.174, -0.696]
```

The exact same process is performed independently for `love`, `machine`, `learning`, and every other token.

### LayerNorm Does Two Things

**Center**

Subtract the mean:

$$
x_i-\mu
$$

**Scale**

Divide by standard deviation:

$$
\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

So LayerNorm is:

**centering + scaling**

### Types of Layer Norm

1. Post Layer Norm (Add & Norm)
2. Pre Layer Norm

#### Difference

| Feature | Post-Layer Norm (Post-LN) | Pre-Layer Norm (Pre-LN) |
| :--- | :--- | :--- |
| **Normalization Order** | Normalization happens **after** the residual addition. | Normalization happens **before** the sub-layer function. |
| **Formula** | $x_{l+1} = \text{LayerNorm}(x_l + \text{SubLayer}(x_l))$ | $x_{l+1} = x_l + \text{SubLayer}(\text{LayerNorm}(x_l))$ |
| **Visual Flow** | Input → SubLayer → Add → Norm → Output | Input → Norm → SubLayer → Add → Output |

### Why Post-LN Is Avoided

- In Post-LN, the layers accumulate/add gradients to very high values, and after normalization and during backpropagation, the gradients diminish to almost nothing. This is the vanishing gradients problem.
- Due to this issue, we are forced to use a warm-up phase. So in general, Pre-LN is kind of risky during training as gradients can either explode or vanish.
- LayerNorm sits on the main path, so the gradient flowing from the loss back to early layers must pass through two LayerNorms per layer (after attention and after the MLP).

For:

$$
x_{l+1} = \text{LayerNorm}(x_l + \text{SubLayer}(x_l))
$$

**What it does:** The model processes the data through the attention layer (SubLayer), adds it to the original data ($\mathbf{x}_l$), and then normalizes the entire combined result at the very end.

### Why Pre-LN Has Become the Modern Standard

For:

$$
x_{l+1} = x_l + \text{SubLayer}(\text{LayerNorm}(x_l))
$$

**What it does:** The model takes your text data ($\mathbf{x}_l$), cleans it up (LayerNorm), processes it through the attention layer (SubLayer), and then adds it back to the original data. So the original data is preserved here.

### Why Are Residual Connections Used Hand in Hand With LayerNorm?

- **The Problem:** In a standard deep network without skips, the error signal gets multiplied by the layer weights over and over. If those weights are small, the signal shrinks exponentially. By the time the error reaches the first few layers, it becomes zero (Vanishing Gradient). The model stops learning.
- **The Solution:** This is why residual connections are used. The error can travel back during backpropagation freely, while we do the "Add" part with the residual connection.

```text
          ┌──────────────── skip path (identity) ───────────────┐
x ────────┤                                                     ├──(+)──► y
          └── LN ──► Attention/MLP ──► (F, the branch) ─────────┘
```
