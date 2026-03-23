# Transformer Attention Visualization

A comprehensive toolkit for visualizing and exploring attention mechanisms in transformer models (BERT). This project provides multiple ways to analyze and understand how BERT's attention heads work across different layers and tokens.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Requirements](#requirements)
- [License](#license)

## 🎯 Overview

This project enables deep exploration of transformer attention patterns by extracting attention weights from pre-trained BERT models and visualizing them using interactive and static heatmaps. Understand which tokens attend to which other tokens and how attention patterns vary across layers and heads.

**Key capabilities:**
- Extract attention matrices from BERT (12 layers × 12 attention heads)
- Generate static visualizations of attention patterns
- Interactive exploration of attention across layers and heads
- Analyze average attention from specific tokens
- Visualize all 12 heads in a single layer simultaneously

## ✨ Features

### 1. **Attention Extraction** (`extract_attention.py`)
   - Loads pre-trained BERT model
   - Tokenizes input sentences
   - Extracts raw attention weights for all 12 layers
   - Returns tensor data for downstream analysis
   - Automatically downloads model (~440MB on first run)

### 2. **Single Head Visualization** (`plot_heatmap.py`)
   - Generates high-quality heatmap for a specific layer and attention head
   - Color gradient: Yellow (low attention) → Red (high attention)
   - Axes labeled with query and key tokens
   - Saves output as PNG image (150 DPI)
   - Customizable layer and head selection

### 3. **All Heads Visualization** (`plot_all_heads.py`)
   - Displays all 12 attention heads in a single layer (3×4 grid)
   - Compact visualization for comparing head behavior
   - Saves high-quality output (130 DPI)
   - Perfect for identifying specialized attention patterns

### 4. **Token Attention Analysis** (`token_attention.py`)
   - Plots attention from a specific token to all other tokens
   - Averages across all 12 heads in selected layer
   - Bar chart visualization with attention weights
   - Color-coded query token for easy identification
   - Displays precise weights on each bar

### 5. **Interactive Explorer** (`explorer.py`)
   - Real-time navigation through layers and heads
   - Keyboard controls:
     - `←` / `→` : Switch between attention heads
     - `↑` / `↓` : Switch between layers
   - Live heatmap updates
   - Perfect for exploratory analysis

## 📁 Project Structure

```
Transformer/
├── extract_attention.py       # Core module: extract BERT attention
├── plot_heatmap.py           # Single head visualization
├── plot_all_heads.py         # All heads in one layer
├── token_attention.py        # Token-specific attention analysis
├── explorer.py               # Interactive attention explorer
├── attention_heatmap.png     # Example output
└── README.md                 # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Swatijha-22/LLM-Project---Message-history-List.git
   cd Transformer
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch matplotlib seaborn transformers
   ```

   Or install all at once:
   ```bash
   pip install torch matplotlib seaborn transformers
   ```

## 📖 Usage

### Basic Usage

```python
from extract_attention import get_attention
from plot_heatmap import plot_head

# Extract attention from a sentence
sentence = "The river bank was flooded after the storm."
tokens, attentions = get_attention(sentence)

# Visualize attention for layer 5, head 0
plot_head(tokens, attentions, layer=5, head=0)
```

### Running Scripts

#### 1. **Single Head Heatmap**
```bash
python plot_heatmap.py
```
Output: `attention_heatmap.png` - Visualizes Layer 6, Head 1

#### 2. **All Heads in a Layer**
```bash
python plot_all_heads.py
```
Output: `all_heads_layer6.png` - 3×4 grid of all 12 heads

#### 3. **Token Attention Analysis**
```bash
python token_attention.py
```
Output: `token_attn_bank.png` - Attention from "bank" token to all others

#### 4. **Interactive Explorer**
```bash
python explorer.py
```
- Opens interactive window
- Use arrow keys to navigate layers/heads
- Press `q` to quit

## 💡 Examples

### Example 1: Analyze Ambiguous Words
```python
from extract_attention import get_attention
from plot_heatmap import plot_head

# Test ambiguous word "bank"
sentence = "The river bank was flooded after the storm."
tokens, attn = get_attention(sentence)

# Check which words attend to "bank"
plot_head(tokens, attn, layer=8, head=3)
```

### Example 2: Compare Multiple Heads
```python
from extract_attention import get_attention
from plot_all_heads import plot_all_heads

sentence = "The quick brown fox jumps over the lazy dog"
tokens, attentions = get_attention(sentence)

# Visualize all heads in layer 10
plot_all_heads(tokens, attentions, layer=10)
```

### Example 3: Token Importance
```python
from extract_attention import get_attention
from token_attention import plot_token_attention

sentence = "Google announced new AI features today"
tokens, attentions = get_attention(sentence)

# See what "announced" pays attention to
plot_token_attention(tokens, attentions, query_token='announced', layer=6)
```

### Example 4: Custom Sentences
```python
# Try your own sentences!
my_sentences = [
    "The cat sat on the mat while the dog played",
    "She left the bank to go to the river bank",
    "Visiting relatives can be tedious",
]

for sentence in my_sentences:
    tokens, attentions = get_attention(sentence)
    plot_head(tokens, attentions, layer=6, head=2)
```

## 🔧 Customization

### Adjust heatmap colors in `plot_heatmap.py`
```python
# Change 'YlOrRd' to other matplotlib colormaps:
# 'RdYlBu', 'viridis', 'coolwarm', 'YlGnBu', etc.
sns.heatmap(..., cmap='RdYlBu', ...)
```

### Change tokenizer/model in `extract_attention.py`
```python
# Use different BERT models:
model_name = 'bert-base-cased'           # Case-sensitive
model_name = 'bert-large-uncased'        # Larger model
model_name = 'distilbert-base-uncased'   # Faster, smaller
```

### Adjust figure size
```python
# In any plot file:
fig, ax = plt.subplots(figsize=(12, 10))  # Increase from default
```

## 📊 Understanding Attention

**What is Attention?**
- Each attention head learns different patterns of which tokens to focus on
- Represented as a matrix: rows = query tokens, columns = key tokens
- Values range from 0 (no attention) to 1 (full attention)

**Visualization Guide:**
- 🟡 Yellow = Low attention weight (limited focus)
- 🔴 Red = High attention weight (strong focus)
- Diagonal usually shows high attention (self-attention)
- Off-diagonal patterns reveal token relationships

## 📋 Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥1.9.0 | Tensor operations |
| transformers | ≥4.10.0 | BERT model & tokenizer |
| matplotlib | ≥3.3.0 | Plotting library |
| seaborn | ≥0.11.0 | Statistical visualization |

## 🎓 Learning Resources

### Understanding Transformers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide

### BERT Specifics
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Exbert: A Visual Analysis Tool to Explore Learned Representations](https://exbert.net/)

## 🐛 Troubleshooting

### Model download fails
```python
# Set cache directory manually in extract_attention.py
import torch
torch.hub.set_dir('/path/to/cache')
```

### Graphics display issues (headless environment)
```python
# Uncomment plt.show() in explorer.py or set backend:
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

### Memory issues with large models
```python
# Use lighter model:
model_name = 'distilbert-base-uncased'  # ~40MB vs 440MB
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Submit issues and bug reports
- Propose new features
- Create pull requests with improvements
- Share visualizations and findings

## 📧 Contact

Created by **Swati Jha** - [GitHub Profile](https://github.com/Swatijha-22)

Feel free to reach out with questions, suggestions, or feedback!

---

**Happy exploring! 🎉**
