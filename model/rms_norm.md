* Normalization is a technique of scaling the values of activations, so that convergence happens more smoothly, and when we backpropagate & perform gradient descent, it is smoother.

* There are several different type of normalization techniques and it all differs based on 'which dimension we are normalizing on'.

* Traditionally in Norm techniques, we use γ and β as the learnable parameters in the expression where γ is the scaling parameter and β is the shifting parameter.

## Batch Norm

* Traditional CNNs used BatchNorm which basically uses all the other values in a batch of values to normalize a particular value. 

* Mathematically:   $$ \hat{x} =  \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}} $$   
* then: $$ y_i = γ_i(\hat{x_i}) + β_i $$	​


* for an example of a Fully connected NN: 

Batch × Features
4 × 3
        f1   f2   f3
x1      2    5    8    ↓
x2      4    7    6    ↓
x3      6    9    4    ↓
x4      8    3    2    ↓

* BatchNorm normally calculates statistics for each feature across the batch


### Why BatchNorm isn't ideal for Transformers

Transformers deal with sequences.

Suppose:

"I love machine learning"

gets represented as:

                                            hidden dimensions
                                                ↓
I love machine learning a lot → [0.2, 1.3, -0.5, 0.7, 0.2, -0.1]
I like cats                   → [0.7, 0.2,  1.1, ___, ___,  ___]
I am   a       human          → [0.9, 0.4,  0.3, 0.4, ___,  ___]

Padding/masking becomes involved.

when batchnorm sees a mask, it doesn't ignore it, it is accounted as zero and the overall normalization.



## Layer Norm

* Was used by transformers until something better appeared
* the fundamental difference is Batch norm is ↓ and Layer norm is →.
* For a tensor of shape (batch, seq_len, d_model), LayerNorm computes mean and variance across the d_model numbers of each individual token. It does not normalize across the sentence. Each token is handled independently, which is why padding and sequence length don't matter to it.

* formula is exact same as batchnorm. only the direction along which we normalize differ.

* Mathematically:   $$ \hat{x} =  \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}} $$  
* then: $$ y_i = γ_i(\hat{x_i}) + β_i $$	

* Importantly: The normalization of one token does not depend on other tokens or other examples in the batch.

* LayerNorm does two things:

Center

Subtract the mean:

$$ x_i-\mu $$
Scale

Divide by standard deviation:

$$ \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}} $$

So LayerNorm is:

centering + scaling
	​
### Types of Layer Norm:
1. Post Layer Norm (Add&Norm)
2. Pre Layer Norm

#### Difference: 

| Feature | Post-Layer Norm (Post-LN) | Pre-Layer Norm (Pre-LN) |
| :--- | :--- | :--- |
| **Normalization Order** | Normalization happens **after** the residual addition. | Normalization happens **before** the sub-layer function. |
| **Formula** | \(x_{l+1} = \text{LayerNorm}(x_l + \text{SubLayer}(x_l))\) | \(x_{l+1} = x_l + \text{SubLayer}(\text{LayerNorm}(x_l))\) |
| **Visual Flow** | Input → SubLayer → Add → Norm → Output | Input → Norm → SubLayer → Add → Output |


### Why Post-LN is avoided:

* In Post-LN, the layers accumulate/add gradients to very high values, and after normalization and during backpropagation, the gradients diminish to almost nothing. This is the vanishing gradients problem.
* Due to this issue, we are forced to use a warm up phase. So in general, Pre-LN is kind of risky during training as gradients can either explode or vanish.
* LayerNorm sits on the main path, so the gradient flowing from the loss back to early layers must pass through two LayerNorms per layer (after attention and after the MLP)

* \(x_{l+1} = \text{LayerNorm}(x_l + \text{SubLayer}(x_l))\): 
What it does: The model processes the data through the attention layer (SubLayer), adds it to the original data (\[\mathbf{x}_{\mathbf{l}}\]), and then normalizes the entire combined result at the very end. 


### Why Pre-LN Has Become the Modern Standard:
* \(x_{l+1} = x_l + \text{SubLayer}(\text{LayerNorm}(x_l))\):
What it does: The model takes your text data (\[\mathbf{x}_{\mathbf{l}}\]), cleans it up (LayerNorm), processes it through the attention layer (SubLayer), and then adds it back to the original data. So the original data is preserved here. 


### Why are Residual connections used as a term hand in hand with LayerNorm:
* The Problem: In a standard deep network without skips, the error signal gets multiplied by the layer weights over and over. If those weights are small, the signal shrinks exponentially. By the time the error reaches the first few layers, it becomes zero (Vanishing Gradient). The model stops learning. 
* The Solution: THis is why residual connections are used. The error can travel back during backpropagation freely, while we do the 'Add' part with the residual connection.

*           ┌──────────────── skip path (identity) ───────────────┐
x ──────────┤                                                     ├──(+)──► y
            └── LN ──► Attention/MLP ──► (F, the branch) ─────────┘
