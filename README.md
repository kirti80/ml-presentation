# ML Presentation

A professional, comprehensive presentation on **Backpropagation and Optimization Algorithms** built with [Marp](https://marp.app/) — ready to export directly to PowerPoint (PPTX), PDF, or interactive HTML.

---

## 📁 Files

| File | Description |
|------|-------------|
| `backprop-optimization.md` | Main Marp presentation (35+ slides) |
| `README.md` | Setup and export instructions |

---

## 📋 Presentation Contents

1. **Neural Network Training Overview**
2. **Backpropagation**
   - Forward Pass
   - Loss Computation
   - Chain Rule & Gradients
   - Backward Pass
   - Gradient Flow & Computational Graphs
   - Vanishing / Exploding Gradients
3. **Optimization Algorithms**
   - Batch / SGD / Mini-Batch Gradient Descent
   - Momentum & Nesterov (NAG)
   - AdaGrad, RMSprop
   - Adam, AdamW, Nadam
4. **Comparison Tables & Best Practices**
   - When to use which optimizer
   - Learning rate schedules
   - Hyperparameter tuning guide
5. **References & Quick Start**

---

## 🚀 Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v14 or later)

### Install Marp CLI

```bash
npm install -g @marp-team/marp-cli
```

---

## 📤 Export Instructions

### Export to PowerPoint (PPTX)

```bash
marp backprop-optimization.md --pptx -o backprop-optimization.pptx
```

### Export to PDF

```bash
marp backprop-optimization.md --pdf -o backprop-optimization.pdf
```

### Export to HTML (interactive, shareable)

```bash
marp backprop-optimization.md --html -o backprop-optimization.html
```

### Live Preview in Browser

```bash
marp --watch backprop-optimization.md
```

This opens a live-reloading browser window — edit the Markdown and see changes instantly.

---

## 🎨 Customizing the Theme

The presentation uses a custom dark theme defined in the Marp frontmatter at the top of `backprop-optimization.md`.

### Change the Color Scheme

Edit the `style` block in the frontmatter:

```yaml
---
marp: true
theme: default
style: |
  section {
    background-color: #1a1a2e;   /* slide background */
    color: #eaeaea;               /* default text color */
  }
  h1 { color: #e94560; }         /* heading color */
---
```

### Use a Built-in Marp Theme

Replace `theme: default` with any of:
- `theme: gaia`
- `theme: uncover`

### Use a Custom Theme File

```bash
marp --theme my-theme.css backprop-optimization.md --pptx -o output.pptx
```

---

## 🔧 Advanced Export Options

```bash
# Set slide size to 16:9 widescreen
marp --size 16:9 backprop-optimization.md --pptx -o output.pptx

# Enable HTML tags in Markdown (for advanced layouts)
marp --html backprop-optimization.md -o output.html

# Use a specific engine version
marp --engine @marp-team/marp-core backprop-optimization.md --pptx
```

---

## 📖 Marp Resources

- [Marp Official Site](https://marp.app/)
- [Marp CLI Documentation](https://github.com/marp-team/marp-cli)
- [Marp Core (themes & directives)](https://github.com/marp-team/marp-core)
- [VS Code Marp Extension](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode) — preview slides in VS Code

---

## 📚 Topics Covered

### Backpropagation
- Forward and backward pass mechanics
- Chain rule derivation
- Layer-wise gradient formulas
- Vanishing and exploding gradient problems

### Optimizers
- SGD, Mini-Batch GD, Momentum, NAG
- AdaGrad, RMSprop
- Adam, AdamW, Nadam
- Convergence comparison and selection guide

---

## 🤝 Contributing

Feel free to open a PR to:
- Add new optimizer slides (e.g., Lion, Sophia)
- Improve diagrams or add Mermaid visuals
- Translate the presentation
