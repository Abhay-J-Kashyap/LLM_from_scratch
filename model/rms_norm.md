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

Transformers deal with sequences.

Suppose:

> "I love machine learning"

gets represented as:

```text
                                            hidden dimensions
                                                ↓

I love machine learning a lot → [0.2, 1.3, -0.5, 0.7, 0.2, -0.1]
I like cats                   → [0.7, 0.2,  1.1, ___, ___,  ___]
I am   a       human          → [0.9, 0.4,  0.3, 0.4, ___,  ___]
```

Padding/masking becomes involved.

When BatchNorm sees a mask, it doesn't ignore it. It is accounted for as zero in the overall normalization.

## Layer Norm

- Was used by transformers until something better appeared.
- The fundamental difference is **BatchNorm is ↓ and LayerNorm is →**.
- For a tensor of shape `(batch, seq_len, d_model)`, LayerNorm computes mean and variance across the `d_model` numbers of each individual token. It does not normalize across the sentence. Each token is handled independently, which is why padding and sequence length don't matter to it.
- The formula is exactly the same as BatchNorm. Only the direction along which we normalize differs.
- Mathematically: $\hat{x} = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}$
- Then: $y_i = \gamma_i(\hat{x_i}) + \beta_i$

Importantly, the normalization of one token does not depend on other tokens or other examples in the batch.

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
