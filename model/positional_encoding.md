# Sinusoidal Positional Encoding in Transformers

## 1. Why do we need positional encoding?

Let's first understand the problem.

In a Transformer, self-attention is used to model relationships between tokens in a sequence. Unlike an RNN, the original Transformer does not use a recurrent structure to process tokens one after another. Self-attention can process all input positions in parallel within a layer.

But this creates an important problem: **how does the model know the order of the tokens?**

Consider these two sentences:

- The dog chased the cat.
- The cat chased the dog.

The words are almost identical, but changing their order changes the meaning.

Self-attention therefore needs some form of positional information to distinguish these arrangements. This is why we introduce *positional encoding*.

## 2. Why not simply assign positions 1, 2, 3, ...?

Our first thought might be to assign a numerical position to each token:

$$
1, 2, 3, \ldots, n
$$

This immediately tells us the order of the tokens. However, directly using these raw scalar values has limitations.

First, the values are unbounded as sequence length increases.

Second, the numerical difference between positions does not automatically provide the model with a convenient, multidimensional representation of positional relationships.

Third, a raw position index does not by itself provide a rich collection of features describing position at different scales.

Notice that the problem is not that integer positions are inherently bad. We can calculate the relative distance between positions by subtraction. Instead, we want a representation that is bounded, changes smoothly as its input changes, and makes relative positional relationships easier for the model to learn.

This leads us to **sinusoidal positional encoding**.

## 3. Why use sine and cosine?

Sine and cosine have useful properties for representing positions:

- Their values are bounded between $-1$ and $1$.
- They change smoothly as their inputs change.
- Their periodic structure gives us a useful mathematical relationship between different positions.

Let's start with just one function:

$$
PE(p) = \sin(p)
$$

For tokens at positions $1$, $2$, and $3$, we get:

$$
\sin(1),\quad \sin(2),\quad \sin(3)
$$

This gives each position a numerical feature.

However, one sinusoidal feature is not enough to provide a rich representation of position. Also, because sine is periodic, its values describe phase modulo its period when considered over real-valued inputs.

So what can we do?

## 4. Why do we use both sine and cosine?

We create a two-dimensional positional vector for each position:

$$
PE(p) =
\begin{bmatrix}
\sin(p) \\
\cos(p)
\end{bmatrix}
$$

Now we have a sine value and a cosine value for the same position.

Geometrically, these two values represent a point on the unit circle:

$$
\sin^2(p) + \cos^2(p) = 1
$$

As the position changes, this point moves around the circle.

Using both functions gives us the phase of the sinusoid through a pair of values, rather than relying on a single sine value.

However, this is still only a two-dimensional positional representation using one frequency. We want to represent position across several different scales.

## 5. The important idea: multiple frequencies

Imagine that we use several sine-cosine pairs:

$$
\begin{aligned}
&\sin(p),\quad \cos(p) \\
&\sin(p/10),\quad \cos(p/10) \\
&\sin(p/100),\quad \cos(p/100)
\end{aligned}
$$

These are only illustrative examples, not the exact formula from the original Transformer paper.

Notice what happens when we divide the position by a larger number.

Consider a sine wave:

$$
y = \sin(\omega p)
$$

Here, $\omega$ is the angular frequency.

When $\omega$ is large, the wave changes rapidly as the position increases. When $\omega$ is small, the wave changes more slowly.

The wavelength is:

$$
\lambda = \frac{2\pi}{\omega}
$$

Therefore, decreasing the frequency increases the wavelength. Dividing the position by a larger number has the same effect as decreasing the frequency.

This gives us two types of positional features:

- **High-frequency features:** change rapidly and capture fine-grained differences between nearby positions.
- **Low-frequency features:** change slowly and capture patterns across larger positional distances.

We can think of this as a collection of clocks running at different speeds. Some distinguish nearby positions, while others track changes more gradually over long sequences.

But how do we choose the exact frequencies?

## 6. The formula used in the original Transformer paper

The original paper, [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), uses the following equations:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

Here:

- $pos$ is the position of the token.
- $i$ identifies the sine-cosine pair.
- $d_{\text{model}}$ is the dimensionality of the Transformer model.
- $2i$ identifies the dimension containing sine.
- $2i+1$ identifies the dimension containing cosine.

We can rewrite the equations using an angular frequency:

$$
\omega_i = \frac{1}{10000^{2i/d_{\text{model}}}}
$$

Then the equations become:

$$
PE_{(pos, 2i)} = \sin(\omega_i pos)
$$

$$
PE_{(pos, 2i+1)} = \cos(\omega_i pos)
$$

Now the pattern becomes much easier to understand: **every pair uses the same position but a different frequency.**

### What if the embedding has 512 dimensions?

Suppose:

$$
d_{\text{model}} = 512
$$

The positional encoding must also have 512 dimensions.

Since every frequency contributes one sine value and one cosine value, we have:

$$
\frac{512}{2} = 256
$$

So there are 256 sine-cosine pairs, corresponding to $i = 0, 1, \ldots, 255$.

For $i = 0$, the denominator is:

$$
10000^0 = 1
$$

The first pair is therefore:

$$
\sin(pos),\quad \cos(pos)
$$

For $i = 1$, the denominator is:

$$
10000^{2/512}
$$

The next pair is:

$$
\sin\left(\frac{pos}{10000^{2/512}}\right),\quad
\cos\left(\frac{pos}{10000^{2/512}}\right)
$$

We continue this process for all 256 pairs.

The denominators increase geometrically, rather than following the sequence $1, 2, 3, \ldots, 256$. Consequently, the wavelengths also increase geometrically.

The result is a 512-dimensional vector containing sinusoidal features at 256 different frequencies.

> **Important correction:** In the original formula, we do not divide by $1, 2, 3, \ldots, 256$. We use the exponentially spaced scale $10000^{2i/d_{\text{model}}}$.

## 7. An interesting analogy: binary encoding

Let's compare this with binary numbers.

Consider the binary representation of an integer. The least significant bit changes very frequently, while more significant bits change less frequently.

For example:

| Decimal | Binary |
| ---: | :--- |
| 0 | `000` |
| 1 | `001` |
| 2 | `010` |
| 3 | `011` |
| 4 | `100` |
| 5 | `101` |
| 6 | `110` |
| 7 | `111` |

The rightmost bit changes every increment. The middle bit changes more slowly, and the leftmost bit changes only when the lower bits complete their cycles.

This is similar to the intuition behind using different sinusoidal frequencies:

- High frequencies change quickly.
- Low frequencies change slowly.

The important difference is that binary encoding uses discrete bits that switch between `0` and `1`, whereas sinusoidal positional encoding produces smooth, continuous-valued features.

The analogy is useful for understanding the different rates of change, but the two encodings are not mathematically equivalent.

### Visualizing the positional encoding patterns

When positional encodings are plotted as a heatmap, each column corresponds to a vector dimension and each row corresponds to a token position. The different frequencies create visible bands and wave-like patterns: some dimensions change rapidly across positions, while others change slowly.

These two articles include useful visual explanations:

- [Positional Encodings: Main Approaches — Medium (Mantis NLP)](https://medium.com/mantisnlp/positional-encodings-i-main-approaches-bd1199d6770d)
- [Designing Positional Encodings — Hugging Face (animation)](https://huggingface.co/blog/designing-positional-encoding#:~:text=The%20above%20animation%20visualizes%20our%20position%20embedding%20if%20each%20component%20is%20alternatively%20drawn%20from)

## 8. How do we combine the positional vector with the token embedding?

Suppose a token is converted into an embedding vector:

$$
E_{\text{token}} \in \mathbb{R}^{512}
$$

We calculate its positional encoding:

$$
PE(pos) \in \mathbb{R}^{512}
$$

Now, how do we provide both pieces of information to the first Transformer layer?

One possible approach is concatenation:

$$
X_{\text{pos}} = [E_{\text{token}}; PE(pos)]
$$

But this produces a vector of 1024 dimensions. That increases the input width, and subsequent projection layers may need more parameters and computation if we retain the same output dimensions.

The original Transformer instead adds the two vectors element by element:

$$
\boxed{X_{\text{pos}} = E_{\text{token}} + PE(pos)}
$$

In other words, the first embedding component is added to the first positional component, the second to the second, and so on.

Because both vectors have the same dimensionality, the resulting vector remains 512-dimensional.

The Transformer can now process a representation containing both token information and positional information without increasing the model dimension.

In the original paper, the token embeddings are scaled by $\sqrt{d_{\text{model}}}$ before positional encoding is added. More precisely, the input is:

$$
X_{\text{pos}} = \sqrt{d_{\text{model}}}\,E_{\text{token}} + PE(pos)
$$

In simplified explanations and many implementations, the key operation is presented simply as embedding plus positional encoding.

Concatenation is not mathematically wrong; it is simply a different architectural choice. The original Transformer uses addition so the model dimension does not grow.

## 9. The main question: how does this help the model understand relative positions?

This is the most important mathematical property of sinusoidal positional encoding.

So far, we have assigned an encoding to each absolute position. But in language, the distance between tokens is often important.

For example, consider a token at position $p$ and another token at position $p+k$.

Can we express the positional encoding at $p+k$ in terms of the positional encoding at $p$, using a transformation that depends only on $k$?

**Yes. This is where the sine-cosine pair becomes especially useful.**

### Step 1: Start with one sine-cosine pair

Consider the two-dimensional positional vector associated with frequency $\omega_i$:

$$
\mathbf{e}_i(p) =
\begin{bmatrix}
\sin(\omega_i p) \\
\cos(\omega_i p)
\end{bmatrix}
$$

We want to move from position $p$ to position $p+k$.

Our target is:

$$
\mathbf{e}_i(p+k) =
\begin{bmatrix}
\sin(\omega_i(p+k)) \\
\cos(\omega_i(p+k))
\end{bmatrix}
$$

Let $k$ represent the relative displacement between the positions. We want a matrix that performs this shift.

### Step 2: Introduce a general matrix

Let's start with a general $2\times2$ matrix containing unknown coefficients:

$$
M =
\begin{bmatrix}
u_1 & v_1 \\
u_2 & v_2
\end{bmatrix}
$$

We want:

$$
M\mathbf{e}_i(p) = \mathbf{e}_i(p+k)
$$

Expanding the left-hand side gives:

$$
\begin{bmatrix}
u_1\sin(\omega_i p)+v_1\cos(\omega_i p) \\
u_2\sin(\omega_i p)+v_2\cos(\omega_i p)
\end{bmatrix}
$$

We need to choose the four coefficients so that this expression equals the encoding at $p+k$.

### Step 3: Use the trigonometric addition identities

Recall these two identities:

$$
\sin(a+b) = \sin(a)\cos(b) + \cos(a)\sin(b)
$$

$$
\cos(a+b) = \cos(a)\cos(b) - \sin(a)\sin(b)
$$

Set:

$$
a = \omega_i p, \qquad b = \omega_i k
$$

Applying the identities to our target vector gives:

$$
\begin{aligned}
\sin(\omega_i(p+k)) ={}& \sin(\omega_i p)\cos(\omega_i k) \\
&+ \cos(\omega_i p)\sin(\omega_i k)
\end{aligned}
$$

Similarly:

$$
\begin{aligned}
\cos(\omega_i(p+k)) ={}& \cos(\omega_i p)\cos(\omega_i k) \\
&- \sin(\omega_i p)\sin(\omega_i k)
\end{aligned}
$$

Now compare these equations with the two rows of our unknown matrix.

### Step 4: Find the coefficients

For the first row, we need:

$$
u_1 = \cos(\omega_i k), \qquad
v_1 = \sin(\omega_i k)
$$

For the second row, we need:

$$
u_2 = -\sin(\omega_i k), \qquad
v_2 = \cos(\omega_i k)
$$

Substituting these coefficients into our matrix gives:

$$
\boxed{
M_i(k) =
\begin{bmatrix}
\cos(\omega_i k) & \sin(\omega_i k) \\
-\sin(\omega_i k) & \cos(\omega_i k)
\end{bmatrix}
}
$$

Therefore, we have derived the relationship:

$$
\boxed{\mathbf{e}_i(p+k) = M_i(k)\mathbf{e}_i(p)}
$$

Notice the key point: **the matrix depends on $k$, the relative displacement, and $\omega_i$, the frequency. It does not depend on the starting position $p$.**

For example, shifting from position 10 to position 13 uses the same matrix as shifting from position 20 to position 23, because the displacement is $k=3$ in both cases.

This matrix is a rotation-type transformation for the sine-cosine pair. With the vector ordered as $[\sin(\omega_i p),\cos(\omega_i p)]^T$, it has the sign arrangement shown above.

### Step 5: Extend the idea to the complete positional vector

So far, we have shown this property for one sine-cosine pair. But our actual positional encoding contains many pairs with different frequencies.

For a model dimension of $d_{\text{model}}=512$, we have 256 pairs. Each pair has its own transformation matrix:

$$
M_0(k), M_1(k), \ldots, M_{255}(k)
$$

We can combine these into one block-diagonal matrix:

$$
\mathcal{M}(k) =
\begin{bmatrix}
M_0(k) & 0 & \cdots & 0 \\
0 & M_1(k) & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & M_{255}(k)
\end{bmatrix}
$$

Here, each diagonal block is a $2\times2$ matrix, and the off-diagonal blocks are zero matrices. The displayed matrix is schematic: each block acts on its corresponding sine-cosine dimensions.

This gives us the relationship for the complete positional encoding:

$$
\boxed{PE(p+k) = \mathcal{M}(k)PE(p)}
$$

This is the result we were looking for.

We can express the positional encoding at a shifted position as a linear transformation of the positional encoding at the original position. The transformation depends on the displacement rather than the absolute starting position.

An equivalent intuition is that each sine-cosine pair rotates through an angle $\omega_i k$ when we shift by $k$ positions. The higher-frequency pairs rotate more rapidly, while the lower-frequency pairs rotate more slowly.

### Step 6: How does self-attention benefit from this?

Self-attention calculates relationships between token representations using queries, keys, and values:

$$
\mathrm{Attention}(Q,K,V) =
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The queries and keys are learned projections of the representations that already contain positional information.

Because sinusoidal positional encodings have a predictable mathematical relationship under a fixed shift, the model can learn transformations and attention patterns that make use of relative distances between tokens.

This does not mean the model is explicitly handed a separate matrix for every distance, or that the transformation converts an arbitrary vector into any arbitrary other vector. Rather, the positional encodings are constructed so that fixed positional shifts have a structured, learnable relationship.

As one further illustration, consider the dot product of two positional vectors for a single frequency, separated by $k$ positions:

$$
\begin{aligned}
\mathbf{e}_i(p)^T\mathbf{e}_i(p+k)
={}& \sin(\omega_i p)\sin(\omega_i(p+k)) \\
&+ \cos(\omega_i p)\cos(\omega_i(p+k)) \\
={}& \cos(\omega_i k)
\end{aligned}
$$

Notice that the final expression depends only on the distance $k$, not the starting position $p$.

This is another way to see how a sine-cosine pair provides a structured relationship between positions. It does not mean that every attention score is determined only by distance, since real attention also depends on token content, learned projections, and other model operations.

## 10. Does the periodic nature of sine cause positions to repeat?

A sine function repeats every $2\pi$, but token positions are integers. Since $2\pi$ is not an integer, exact repetition at different integer positions is not inevitable in ideal real arithmetic. Values can nevertheless become arbitrarily close, and finite-precision computation has limits.

More importantly, sine and cosine are not combined merely to prevent repetition. Their key mathematical advantage is that **shifting the position by a fixed distance can be expressed as a linear transformation of the sine-cosine pair**.

Using multiple frequencies also gives each position a richer pattern of features than a single sinusoid would provide. Still, sinusoidal encodings should not be described as guaranteeing a unique, perfectly distinguishable vector for every possible integer position under all numerical conditions.

## 11. Final summary

We began with a problem: self-attention needs access to token order because it does not inherently provide the sequential ordering mechanism of an RNN.

We introduced sinusoidal positional encoding, which constructs a vector of bounded, smooth features using sine and cosine at multiple frequencies.

The original Transformer uses a geometrically spaced family of frequencies. The positional vector has the same dimensionality as the token embedding, allowing the two vectors to be added element by element.

Most importantly, each sine-cosine pair has the property that shifting a position by a fixed distance is equivalent to applying a linear transformation that depends on that distance. The same property extends to the complete positional vector through a block-diagonal transformation.

**That is the central mathematical insight: sinusoidal positional encoding represents absolute positions in a way that also makes relative positional shifts accessible through structured linear transformations.**

This was the approach used in the original *Attention Is All You Need* paper. Not every modern Transformer uses this exact encoding scheme; other approaches to positional information are also widely used.

## References and further visualization

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017.
- [Positional Encodings: Main Approaches — Medium (Mantis NLP)](https://medium.com/mantisnlp/positional-encodings-i-main-approaches-bd1199d6770d)
- [Designing Positional Encodings — Hugging Face (animation)](https://huggingface.co/blog/designing-positional-encoding#:~:text=The%20above%20animation%20visualizes%20our%20position%20embedding%20if%20each%20component%20is%20alternatively%20drawn%20from)
