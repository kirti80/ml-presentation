---
marp: true
theme: default
paginate: true
backgroundColor: '#1a1a2e'
color: '#eaeaea'
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    background-color: #1a1a2e;
    color: #eaeaea;
  }
  h1 { color: #e94560; font-size: 2.2em; border-bottom: 3px solid #e94560; padding-bottom: 10px; }
  h2 { color: #0f3460; background: #16213e; padding: 8px 16px; border-radius: 6px; }
  h3 { color: #e94560; }
  code { background: #16213e; border-radius: 4px; padding: 2px 6px; color: #a8ff78; }
  pre { background: #16213e; border-left: 4px solid #e94560; padding: 16px; border-radius: 8px; }
  table { width: 100%; border-collapse: collapse; }
  th { background: #0f3460; color: #eaeaea; padding: 8px; }
  td { border: 1px solid #0f3460; padding: 8px; }
  tr:nth-child(even) { background: #16213e; }
  blockquote { border-left: 4px solid #e94560; padding-left: 12px; color: #a8a8c8; }
---

<!-- _class: lead -->
# Backpropagation & Optimization Algorithms

### A Deep Dive into Training Neural Networks

---

## About This Presentation

*From gradient computation to state-of-the-art optimizers*

**Target Audience:** Intermediate to Advanced ML Practitioners

---

## 📋 Table of Contents

1. **Neural Network Training Overview**
2. **Backpropagation**
   - Forward Pass
   - Loss Computation
   - Chain Rule & Gradients
   - Backward Pass
   - Computational Graphs
3. **Optimization Algorithms**
   - Gradient Descent Variants
   - Momentum, RMSprop, Adam, AdamW
   - AdaGrad, Nadam, and more
4. **Comparison, Best Practices & Summary**

---

## 🧠 Neural Network Training: The Big Picture

Training a neural network = finding parameters **W** that minimize a **loss function** ℒ

```
  Input  →  [Layer 1]  →  [Layer 2]  →  [Output]  →  Loss
     ←──────────────── Backpropagation ───────────────
```

**The Training Loop:**

```python
for epoch in range(num_epochs):
    # 1. Forward pass — compute predictions
    y_pred = model(X)
    # 2. Compute loss
    loss = loss_fn(y_pred, y_true)
    # 3. Backward pass — compute gradients
    loss.backward()
    # 4. Update parameters
    optimizer.step()
    optimizer.zero_grad()
```

---

## 🔗 The Neural Network Architecture

```
Input Layer      Hidden Layer 1    Hidden Layer 2    Output Layer
   x₁ ──────────────┐
   x₂ ──────────────┤  h₁¹  h₁²       h₂¹  h₂²        ŷ₁
   x₃ ──────────────┤  h₂¹  h₂²  ───  h₂³  h₂⁴  ───   ŷ₂
   ...              │  ...  ...        ...  ...
   xₙ ──────────────┘
```

**Each neuron computes:**

$$a^{(l)} = \sigma\left(W^{(l)} \cdot a^{(l-1)} + b^{(l)}\right)$$

Where:
- $W^{(l)}$ = weight matrix of layer *l*
- $b^{(l)}$ = bias vector of layer *l*
- $\sigma$ = activation function (ReLU, sigmoid, tanh, …)

---

## ⏩ Step 1: Forward Pass

**Goal:** Compute predictions from inputs layer by layer.

```
x  →  z¹ = W¹x + b¹  →  a¹ = σ(z¹)  →  z² = W²a¹ + b²  →  ŷ = σ(z²)
```

**Layer-by-layer computation:**

```python
def forward(x, weights, biases):
    activations = [x]
    for W, b in zip(weights, biases):
        z = W @ activations[-1] + b     # linear transform
        a = relu(z)                      # activation
        activations.append(a)
    return activations
```

**Key insight:** We cache all intermediate activations $a^{(l)}$ — they are needed in the backward pass!

---

## 📉 Step 2: Loss Computation

**Goal:** Quantify how wrong our predictions are.

### Common Loss Functions

| Task | Loss Function | Formula |
|------|--------------|---------|
| Regression | Mean Squared Error | $\mathcal{L} = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$ |
| Binary Classification | Binary Cross-Entropy | $\mathcal{L} = -\sum y\log\hat{y} + (1-y)\log(1-\hat{y})$ |
| Multi-class | Categorical Cross-Entropy | $\mathcal{L} = -\sum_c y_c \log\hat{y}_c$ |

**Example (MSE):**

$$\mathcal{L}(W, b) = \frac{1}{n}\sum_{i=1}^{n}\left(\hat{y}_i - y_i\right)^2$$

This scalar loss is the starting point for gradient computation.

---

## 🔀 Step 3: The Chain Rule

**Problem:** How does changing $W^{(1)}$ affect the final loss $\mathcal{L}$?

### Chain Rule (single variable)

$$\frac{d\mathcal{L}}{dx} = \frac{d\mathcal{L}}{dz} \cdot \frac{dz}{dx}$$

### Chain Rule (multi-layer network)

$$\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \frac{\partial \mathcal{L}}{\partial a^{(3)}} \cdot \frac{\partial a^{(3)}}{\partial a^{(2)}} \cdot \frac{\partial a^{(2)}}{\partial a^{(1)}} \cdot \frac{\partial a^{(1)}}{\partial W^{(1)}}$$

> The chain rule allows us to decompose the gradient through each layer — this is the core of backpropagation.

---

## ⏪ Step 4: Backward Pass (Backpropagation)

**Goal:** Propagate gradients from the output layer back to every weight.

```
Loss  →  ∂L/∂ŷ  →  ∂L/∂W²  →  ∂L/∂a¹  →  ∂L/∂W¹  →  ∂L/∂x
         ←────────────── Gradient Flow ───────────────────
```

**Gradient computation per layer:**

```python
def backward(loss_grad, activations, weights):
    grads_W, grads_b = [], []
    delta = loss_grad                          # ∂L/∂output
    for W, a_prev in reversed(list(zip(weights, activations[:-1]))):
        dW = delta @ a_prev.T                  # ∂L/∂W = δ · aᵀ
        db = delta.sum(axis=1, keepdims=True)  # ∂L/∂b = δ
        delta = W.T @ delta * relu_grad(a_prev) # propagate back
        grads_W.insert(0, dW)
        grads_b.insert(0, db)
    return grads_W, grads_b
```

---

## 📊 Gradient Flow Through Layers

```
         Forward Pass
  ┌──────────────────────────────────────────────┐
  │  x  →  [W¹,b¹]  →  [W²,b²]  →  [W³,b³]  → ŷ │
  └──────────────────────────────────────────────┘
              ↓ Cache: z¹, a¹, z², a², z³, a³
         Backward Pass
  ┌──────────────────────────────────────────────┐
  │ ∂L/∂W¹ ← ∂L/∂a¹ ← ∂L/∂a² ← ∂L/∂a³ ← ∂L/∂ŷ │
  └──────────────────────────────────────────────┘
```

**Gradient formulas for a layer:**

$$\delta^{(l)} = \left(W^{(l+1)}\right)^T \delta^{(l+1)} \odot \sigma'\left(z^{(l)}\right)$$

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \left(a^{(l-1)}\right)^T$$

$$\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}$$

Where $\odot$ is element-wise (Hadamard) product.

---

## 🕸 Computational Graph Example

**Expression:** $y = (x + w) \cdot v$

```
   x ──┐
        ├──[ + ]── q ──┐
   w ──┘               ├──[ × ]── y
                  v ──┘
```

**Forward:** $q = x + w$, $y = q \cdot v$

**Backward (chain rule):**

$$\frac{\partial y}{\partial w} = \frac{\partial y}{\partial q} \cdot \frac{\partial q}{\partial w} = v \cdot 1 = v$$

$$\frac{\partial y}{\partial v} = q$$

> **PyTorch/TensorFlow** build this graph automatically (autograd). Every operation records how to compute its gradient.

---

## ⚠️ Vanishing & Exploding Gradients

### Vanishing Gradients
- Gradients shrink exponentially as they propagate backward
- Especially problematic with **sigmoid/tanh** activations
- Network early layers learn very slowly

**Fix:** Use **ReLU** activations, **residual connections** (ResNets), **batch normalization**

### Exploding Gradients
- Gradients grow exponentially — training diverges
- Common in RNNs with long sequences

**Fix:** **Gradient clipping**

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 🎯 What is Optimization?

**Goal:** Find weights **W** that minimize the loss function $\mathcal{L}(W)$

$$W^* = \arg\min_W \mathcal{L}(W)$$

**The Loss Landscape:**

```
Loss
 │  ╲         local min
 │   ╲   /╲   /╲
 │    ╲_/  ╲_/  ╲___  ← global min
 └─────────────────── W
```

> Optimization algorithms navigate this landscape using gradients.

**Key challenges:**
- Saddle points & local minima
- Flat regions (plateaus)
- Noisy gradients
- Ill-conditioned curvature

---

## 📐 Batch Gradient Descent

**Idea:** Use the **entire dataset** to compute one gradient update.

$$W \leftarrow W - \eta \cdot \nabla_W \mathcal{L}(W)$$

Where $\eta$ is the **learning rate**.

```python
for epoch in range(epochs):
    grad = compute_gradient(W, X_full, y_full)   # full dataset
    W = W - lr * grad
```

**Pros:**
- ✅ Stable, smooth convergence
- ✅ Guaranteed to reach a local minimum

**Cons:**
- ❌ Very slow for large datasets (one update per full pass)
- ❌ High memory usage
- ❌ Cannot update in an online/streaming setting

---

## 🎲 Stochastic Gradient Descent (SGD)

**Idea:** Use a **single random sample** per gradient update.

$$W \leftarrow W - \eta \cdot \nabla_W \mathcal{L}(W; x_i, y_i)$$

```python
for epoch in range(epochs):
    for x_i, y_i in shuffle(dataset):       # one sample at a time
        grad = compute_gradient(W, x_i, y_i)
        W = W - lr * grad
```

**Pros:**
- ✅ Very fast updates — can handle huge datasets
- ✅ Noisy updates help escape local minima
- ✅ Online learning capable

**Cons:**
- ❌ High variance — path is noisy
- ❌ May not converge without learning rate schedule
- ❌ Cannot exploit vectorized hardware (no batching)

---

## 📦 Mini-Batch Gradient Descent

**Idea:** Use a small **random batch** (16–512 samples) per update.

$$W \leftarrow W - \eta \cdot \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla_W \mathcal{L}(W; x_i, y_i)$$

```python
for epoch in range(epochs):
    for batch in dataloader(batch_size=32):  # mini-batch
        grad = compute_gradient(W, batch)
        W = W - lr * grad
```

**Pros:**
- ✅ Best of both worlds: efficiency + stability
- ✅ GPU-friendly (parallel matrix operations)
- ✅ Less noisy than SGD, faster than full-batch GD

**Cons:**
- ❌ Adds hyperparameter: batch size
- ❌ Still sensitive to learning rate choice

> **Mini-batch GD is the standard** in modern deep learning (what PyTorch's SGD optimizer uses by default).

---

## 🚀 Momentum

**Problem with SGD:** Oscillates in ravine-shaped loss landscapes.

**Solution:** Accumulate a **velocity** vector in the direction of persistent gradient.

$$v_t = \beta v_{t-1} + (1 - \beta) \nabla_W \mathcal{L}$$

$$W \leftarrow W - \eta \cdot v_t$$

```python
v = 0
for grad in gradients:
    v = beta * v + (1 - beta) * grad     # exponential moving average
    W = W - lr * v
```

| Parameter | Typical Value | Role |
|-----------|--------------|------|
| $\eta$ | 0.01 | Learning rate |
| $\beta$ | 0.9 | Momentum coefficient |

**Pros:** ✅ Faster convergence in ravines, dampens oscillations  
**Cons:** ❌ Adds one hyperparameter; can overshoot minimum

---

## 💨 Nesterov Accelerated Gradient (NAG)

**Improvement on Momentum:** Look ahead before computing gradient.

**Standard Momentum:**
$$v_t = \beta v_{t-1} + \eta \nabla_W \mathcal{L}(W)$$

**Nesterov:**
$$v_t = \beta v_{t-1} + \eta \nabla_W \mathcal{L}(W - \beta v_{t-1})$$

$$W \leftarrow W - v_t$$

```
Momentum:  ──→ compute gradient at current W → jump
NAG:       ──→ jump first → compute gradient at future W
```

> NAG "corrects" the step by computing the gradient at the anticipated position.

**Pros:** ✅ Faster convergence than standard momentum  
**Cons:** ❌ More complex gradient computation

---

## 📊 AdaGrad — Adaptive Gradient

**Idea:** Give **infrequent features** larger updates by adapting per-parameter learning rates.

$$G_t = G_{t-1} + \left(\nabla_W \mathcal{L}\right)^2$$

$$W \leftarrow W - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot \nabla_W \mathcal{L}$$

```python
G = 0
for grad in gradients:
    G += grad ** 2                        # accumulate squared gradients
    W -= (lr / (np.sqrt(G) + eps)) * grad
```

**Intuition:** Parameters with large accumulated gradients → smaller learning rate; sparse features → larger learning rate.

**Pros:** ✅ No manual learning rate tuning per parameter; great for sparse data  
**Cons:** ❌ $G_t$ grows monotonically → learning rate shrinks to zero (learning stops)

---

## 🌀 RMSprop

**Fix for AdaGrad:** Use an **exponential moving average** of squared gradients instead of a running sum.

$$E[g^2]_t = \rho \cdot E[g^2]_{t-1} + (1 - \rho) \cdot g_t^2$$

$$W \leftarrow W - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t$$

```python
E_g2 = 0
for grad in gradients:
    E_g2 = rho * E_g2 + (1 - rho) * grad**2   # decaying average
    W -= (lr / (np.sqrt(E_g2) + eps)) * grad
```

| Parameter | Typical Value |
|-----------|--------------|
| $\eta$ | 0.001 |
| $\rho$ | 0.9 |
| $\epsilon$ | 1e-8 |

**Pros:** ✅ Solves AdaGrad's vanishing LR problem; works well for RNNs  
**Cons:** ❌ Still requires tuning $\eta$; no bias correction

---

## ⭐ Adam — Adaptive Moment Estimation

**Best of Momentum + RMSprop.** Maintains both 1st and 2nd moment estimates.

**1st moment (mean):** $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$

**2nd moment (variance):** $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$

**Bias correction:** $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$, $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$

**Update:** $W \leftarrow W - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$

```python
m, v = 0, 0
for t, grad in enumerate(gradients, 1):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    W -= lr * m_hat / (np.sqrt(v_hat) + eps)
```

---

## ⭐ Adam — Parameters & Properties

| Parameter | Default Value | Role |
|-----------|--------------|------|
| $\eta$ | 0.001 | Learning rate |
| $\beta_1$ | 0.9 | 1st moment decay (momentum) |
| $\beta_2$ | 0.999 | 2nd moment decay (variance) |
| $\epsilon$ | 1e-8 | Numerical stability |

**Pros:**
- ✅ Works well out-of-the-box for most tasks
- ✅ Bias correction for early training
- ✅ Adaptive per-parameter learning rates
- ✅ Handles sparse & noisy gradients

**Cons:**
- ❌ May not generalize as well as SGD+Momentum for some vision tasks
- ❌ Can converge to sharp minima (poorer generalization)

---

## 🛡 AdamW — Adam with Weight Decay

**Problem with Adam:** L2 regularization (weight decay) is incorrectly coupled with adaptive gradients.

**In standard Adam (with L2 reg):**
$$g_t = \nabla \mathcal{L} + \lambda W$$

Decay is scaled by adaptive LR — it's not "true" weight decay.

**AdamW decouples weight decay:**
$$W \leftarrow W - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda W\right)$$

```python
# AdamW update
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
W -= lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * W)
```

**Pros:** ✅ Better generalization, proper regularization  
**Cons:** ❌ Additional `weight_decay` hyperparameter

> **AdamW is the default choice** for training Transformers (BERT, GPT, etc.)

---

## 🔮 Nadam — Nesterov + Adam

**Combines** Nesterov momentum with Adam's adaptive learning rates.

**Standard Adam update:**
$$W \leftarrow W - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon} \hat{m}_t$$

**Nadam substitutes the Nesterov update for the momentum term:**
$$W \leftarrow W - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon} \left(\beta_1 \hat{m}_t + \frac{(1-\beta_1)g_t}{1-\beta_1^t}\right)$$

**Intuition:** Look ahead using Nesterov, then scale adaptively.

| Parameter | Default Value |
|-----------|--------------|
| $\eta$ | 0.002 |
| $\beta_1$ | 0.9 |
| $\beta_2$ | 0.999 |
| $\epsilon$ | 1e-8 |

**Pros:** ✅ Slightly faster convergence than Adam in many tasks  
**Cons:** ❌ Marginal improvement; not always worth extra complexity

---

## 🏔 Gradient Descent Convergence Paths

```
Loss Surface (contour view):

  ╔═══════════════════════════════════╗
  ║  ●──────────────────────────────  ║  SGD (noisy zigzag)
  ║  ●═══════════════════════════════ ║  Momentum (smooth curve)
  ║  ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ║  Adam (fast, direct)
  ║                         ★         ║  ← minimum
  ╚═══════════════════════════════════╝

  SGD:      ●→→↑→↓→→↑↓→→ (high variance)
  Momentum: ●══════════→  (smooth)
  Adam:     ●━━━━━━━━━→   (fast, adaptive)
```

**Convergence speed ranking (typical):**
1. 🥇 Adam / AdamW — fastest initial convergence
2. 🥈 Momentum / NAG — good for well-tuned LR
3. 🥉 SGD — slow but often best final generalization
4. Last — Batch GD (too slow on large data)

---

## 📊 Optimizer Comparison Table

| Optimizer | LR Adapt | Momentum | Weight Decay | Best For |
|-----------|----------|----------|--------------|---------|
| SGD | ❌ | ❌ | ✅ (manual) | Vision (fine-tuned) |
| SGD+Momentum | ❌ | ✅ | ✅ | Vision, ResNets |
| AdaGrad | ✅ | ❌ | ❌ | Sparse NLP |
| RMSprop | ✅ | ❌ | ❌ | RNNs, RL |
| Adam | ✅ | ✅ | ⚠️ (coupled) | General purpose |
| AdamW | ✅ | ✅ | ✅ (decoupled) | Transformers, LLMs |
| Nadam | ✅ | ✅ (Nesterov) | ❌ | Fast convergence |
| NAG | ❌ | ✅ (Nesterov) | ❌ | Convex, smooth |

---

## 🔧 Hyperparameter Tuning Guide

### Learning Rate — The Most Important Hyperparameter

```
Too low:  ●──────────────→  (converges slowly, may get stuck)
Just right: ●════════→      (smooth, reaches minimum)
Too high: ● ↗↘↗↘↗↘         (diverges or oscillates)
```

**Learning Rate Schedules:**

```python
# Step decay
lr = lr_init * (decay_rate ** (epoch // step_size))

# Cosine annealing
lr = lr_min + 0.5*(lr_max - lr_min) * (1 + cos(π * t/T))

# Warmup (common for Transformers)
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

---

## 🎛 Batch Size Tuning

**Batch Size Effect:**

| Batch Size | Gradient Noise | Memory | Speed | Generalization |
|------------|---------------|--------|-------|----------------|
| 1 (pure SGD) | Very high | Low | Slow | Often good |
| 32–256 | Medium | Medium | Fast | Good |
| 1024+ | Low | High | Very fast | May overfit |

**Rules of thumb:**
- Start with batch size **32** or **64**
- If training is unstable → decrease batch size
- If training is too slow → increase batch size
- Scale learning rate with batch size: $\eta_{new} = \eta_{base} \times \frac{B_{new}}{B_{base}}$

---

## 🔄 Learning Rate Warmup & Schedules

```
LR
 │   ╱──────────────────\
 │  ╱ warmup              \  cosine decay
 │ ╱                        \
 │╱                            \_____
 └─────────────────────────────────── step
   warmup        training         end
```

**Common Schedules:**

| Schedule | When to Use |
|----------|-------------|
| Constant LR | Simple baselines |
| Step Decay | Standard CNNs (ResNet, VGG) |
| Cosine Annealing | SOTA models, cyclical training |
| Warmup + Cosine | Transformers (BERT, GPT, ViT) |
| OneCycleLR | Fast.ai recommended, superconvergence |

---

## 📌 When to Use Which Optimizer

```
New project / general use?  →  Adam or AdamW
NLP / Transformers?          →  AdamW
Computer Vision (ResNet)?    →  SGD + Momentum + LR schedule
RNNs / LSTMs?                →  RMSprop or Adam
Sparse features (NLP embed)? →  AdaGrad or Adam
RL training?                 →  RMSprop or Adam
Fast prototyping?            →  Adam (lr=1e-3)
Best final accuracy?         →  SGD + Momentum (well-tuned)
```

**Decision tree summary:**

1. Start with **Adam** (lr=1e-3, $\beta_1$=0.9, $\beta_2$=0.999)
2. If using weight decay → switch to **AdamW**
3. For Transformers → **AdamW** with warmup + cosine schedule
4. For vision fine-tuning → **SGD + Momentum** (lr=0.01, momentum=0.9)

---

## ✅ Best Practices

### Gradient Handling
- Always **clip gradients** for RNNs: `clip_grad_norm_(params, 1.0)`
- Use **gradient accumulation** to simulate large batches on limited GPU memory
- Monitor **gradient norms** to detect vanishing/exploding issues

### Regularization with Optimization
- **L2 regularization** → use AdamW (not Adam) to correctly decouple
- **Dropout** — complements Adam well
- **Batch Normalization** — reduces sensitivity to learning rate

### Training Stability
- **Warmup** especially for large models / Transformers
- **Mixed precision** (`torch.cuda.amp`) speeds training without loss of accuracy
- **EMA of weights** for better final model quality

---

## 🧩 Backpropagation: Summary

```
FORWARD PASS:
  x → [W₁,b₁] → a₁ → [W₂,b₂] → a₂ → ... → ŷ → L

BACKWARD PASS:
  ∂L/∂ŷ → ∂L/∂W₂ → ∂L/∂a₁ → ∂L/∂W₁ → ∂L/∂x
```

**Key equations:**

$$\delta^{(L)} = \nabla_{a^{(L)}} \mathcal{L} \odot \sigma'\left(z^{(L)}\right)$$

$$\delta^{(l)} = \left(W^{(l+1)}\right)^T \delta^{(l+1)} \odot \sigma'\left(z^{(l)}\right)$$

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \left(a^{(l-1)}\right)^T, \quad \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}$$

---

## 🏁 Optimizers: Summary

| Algorithm | Key Idea | Signature Formula |
|-----------|----------|-------------------|
| SGD | Follow gradient | $W \leftarrow W - \eta g$ |
| Momentum | Velocity accumulation | $W \leftarrow W - \eta v_t$ |
| AdaGrad | Per-param LR (cumsum) | $W \leftarrow W - \frac{\eta}{\sqrt{G_t}+\epsilon}g$ |
| RMSprop | Per-param LR (EMA) | $W \leftarrow W - \frac{\eta}{\sqrt{E[g^2]_t}+\epsilon}g$ |
| Adam | Momentum + RMSprop | $W \leftarrow W - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t$ |
| AdamW | Adam + proper decay | Adam + $\lambda W$ decoupled |
| Nadam | NAG + Adam | Adam with Nesterov momentum |

---

## 📚 References & Further Reading

- **Rumelhart, Hinton & Williams (1986)** — Backpropagation original paper
- **Kingma & Ba (2015)** — *Adam: A Method for Stochastic Optimization* — [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
- **Loshchilov & Hutter (2019)** — *Decoupled Weight Decay Regularization (AdamW)* — [arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
- **Sutton (1986)** — Nesterov Accelerated Gradient
- **Duchi et al. (2011)** — AdaGrad
- **Goodfellow, Bengio & Courville** — *Deep Learning* (Chapter 8 — Optimization)
- **CS231n** — Stanford course: [cs231n.github.io](https://cs231n.github.io)
- **fast.ai** — Practical Deep Learning: [fast.ai](https://www.fast.ai)

---

<!-- _class: lead -->

# 🎉 Thank You!

## Key Takeaways

- **Backpropagation** = chain rule applied layer by layer
- **Gradients** tell us how to adjust each weight
- **Adam/AdamW** = go-to optimizers for most tasks
- **Learning rate** is the most critical hyperparameter
- Always monitor **gradient norms** and use **schedules**

---

*Slides created with [Marp](https://marp.app/) — export to PPTX, PDF, or HTML*

---

## 🚀 Quick Start with Marp

```bash
# Install Marp CLI
npm install -g @marp-team/marp-cli

# Export to PowerPoint
marp backprop-optimization.md --pptx -o backprop-optimization.pptx

# Export to PDF
marp backprop-optimization.md --pdf -o backprop-optimization.pdf

# Export to HTML (interactive)
marp backprop-optimization.md --html -o backprop-optimization.html

# Preview in browser (live reload)
marp --watch backprop-optimization.md
```
