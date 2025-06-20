# 🤖 PyTorch Text Classification with RoBERTa

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-red?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow?style=for-the-badge)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> 🎓 **Learning Project**: Exploring NLP and PyTorch through multi-label text classification

---

## 👨‍💻 About This Project

Hi! I'm **Nuthan Reddy Surukunti**, a student passionate about AI and Machine Learning. This project represents my journey in learning **PyTorch** and **Natural Language Processing (NLP)** by implementing a state-of-the-art text classification system.

### 🎯 **Learning Goals**
- 📚 Master PyTorch Lightning framework
- 🧠 Understand transformer models (RoBERTa)
- 🔍 Explore multi-label classification
- 📊 Handle imbalanced datasets
- 🚀 Build end-to-end ML pipelines

---

## 📧 **Connect With Me**
- 📫 **Email**: [surkuntinuthanreddy@gmail.com](mailto:surkuntinuthanreddy@gmail.com)
- 🐙 **GitHub**: [@Nuthan-Reddy-Surukunti](https://github.com/Nuthan-Reddy-Surukunti)

---

## 📊 **Dataset: UCC (Unhealthy Comments Corpus)**

This project uses the **UCC dataset** - a high-quality corpus for detecting unhealthy conversation patterns, published in the research paper [*"Six Attributes of Unhealthy Conversation"*](https://arxiv.org/abs/2010.07410).

### 📈 **Dataset Statistics**
| Category | Count | Percentage |
|----------|--------|-----------|
| 💚 Healthy Comments | 40,000+ | ~93% |
| ⚠️ Unhealthy Comments | <3,000 | ~7% |

### 🏷️ **Classification Labels (8 attributes)**
| Label | Description | Emoji |
|-------|-------------|-------|
| `antagonize` | Provocative behavior | 😠 |
| `condescending` | Talking down to others | 🙄 |
| `dismissive` | Disregarding opinions | 🚫 |
| `generalisation` | Broad generalizations | 📊 |
| `generalisation_unfair` | Biased generalizations | ⚖️ |
| `hostile` | Aggressive language | 💢 |
| `sarcastic` | Sarcastic remarks | 😏 |
| `unhealthy` | Overall toxicity flag | ☣️ |

---

## 🏗️ **Technical Architecture**

### 🤖 **Model Pipeline**
```
📝 Text Input
    ↓
🔤 RoBERTa Tokenizer (max_length=128)
    ↓
🧠 DistilRoBERTa-Base Model
    ↓
🔄 Mean Pooling Layer
    ↓
🎯 Classification Head (8 outputs)
    ↓
📊 Multi-label Predictions
```

### ⚙️ **Key Components**
- **🤖 Model**: `distilroberta-base` (lightweight RoBERTa)
- **📦 Framework**: PyTorch Lightning
- **🔧 Optimizer**: AdamW with cosine scheduling
- **📏 Loss**: BCEWithLogitsLoss (multi-label)
- **📊 Metrics**: ROC-AUC, Precision, Recall

---

## 🚀 **Getting Started**

### 📋 **Prerequisites**
```bash
# Core ML Libraries
pip install torch torchvision torchaudio
pip install pytorch-lightning
pip install transformers
pip install torchmetrics

# Data & Visualization
pip install pandas numpy matplotlib seaborn
pip install scikit-learn
```

### 🏃‍♂️ **Quick Start**
1. **📂 Clone & Navigate**
   ```bash
   git clone https://github.com/Nuthan-Reddy-Surukunti/pytorch-text-classification
   cd pytorch-text-classification
   ```

2. **📓 Run Jupyter Notebook**
   ```bash
   jupyter notebook RoBERTa_ucc.ipynb
   ```

3. **🐍 Or Run Python Script**
   ```bash
   python roberta_ucc.py
   ```

---

## 📁 **Project Structure**
```
📦 pytorch-Text_Classification/
├── 📓 RoBERTa_ucc.ipynb     # Main implementation notebook
├── 🐍 roberta_ucc.py        # Python script version
└── 📖 README.md             # This file
```

---

## 🎓 **Learning Highlights**

### 🔍 **What I'm Exploring**
- ✅ **Multi-label vs Multi-class** classification differences
- ✅ **Class imbalance** handling with sampling strategies
- ✅ **Transformer attention** mechanisms
- ✅ **PyTorch Lightning** best practices
- ✅ **HuggingFace** ecosystem integration

### 🧪 **Experiments & Learnings**
- 🔬 **Data Preprocessing**: Tokenization, padding, attention masks
- 📊 **Model Training**: Learning rate scheduling, warmup strategies
- 📈 **Evaluation**: ROC curves, confusion matrices, multi-label metrics
- 🎨 **Visualization**: TensorBoard integration, custom plots

---

## 🎯 **Configuration**
```python
config = {
    '🤖 model_name': 'distilroberta-base',
    '🏷️ n_labels': 8,
    '📦 batch_size': 128,
    '📈 learning_rate': 1.5e-6,
    '🔥 warmup': 0.2,
    '⚖️ weight_decay': 0.001,
    '🔄 epochs': 80
}
```

---

## 🔮 **Future Learning Plans**
- 🚀 **Deploy** model as a web application using Streamlit
- 📊 **Experiment** with other transformer models (BERT, T5, GPT)
- 🎯 **Improve** performance with advanced techniques
- 🌐 **Try** other NLP datasets and tasks
- 📱 **Build** a real-time comment moderation system

---

## 📚 **Learning Resources**
- 📖 [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- 🤗 [HuggingFace Transformers](https://huggingface.co/transformers/)
- 📝 [Original UCC Paper](https://arxiv.org/abs/2010.07410)
- 🎓 [Fast.ai NLP Course](https://course.fast.ai/)

---

## 🤝 **Contributing & Feedback**
I'm always eager to learn! If you have suggestions, improvements, or want to discuss NLP concepts, feel free to reach out:

- 💌 **Email**: [surkuntinuthanreddy@gmail.com](mailto:surkuntinuthanreddy@gmail.com)
- 🐙 **GitHub Issues**: Create an issue for discussions
- 💡 **Ideas Welcome**: Share your thoughts on improvements!

---

## 📜 **License**
This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

### 🌟 **Happy Learning!** 🌟
*Made with ❤️ by Nuthan Reddy Surukunti*

</div>

