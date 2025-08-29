# Hinglish → English Neural Machine Translation

<div align="center">

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.8+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

*A Transformer-based Neural Machine Translation (NMT) model for translating Hinglish (Hindi-English mixed text) into English with transformer architecture implemented from the Attention is all your need research paper.*

</div>

---

## ✨ Features

- 🤖 **Transformer Architecture** - Implemented from the renowned "Attention is All You Need" research paper
- 🔧 **Text Preprocessing** - Advanced tokenization and preprocessing for Hinglish and English text
- 📦 **Pre-trained Model** - Ready-to-use trained model (tracked with Git LFS)
- 📓 **Interactive Notebook** - Complete Jupyter notebook for training and inference

---

## 📊 Dataset

The training dataset is hosted externally due to GitHub size limitations:

🔗 **[Download Dataset from Hugging Face](https://huggingface.co/datasets/nateraw/english-to-hinglish/viewer/default/train?views%5B%5D=train)**

> ⚠️ **Important:** Please download the dataset before running the notebook or scripts.

---

## 🧠 Model

The trained Transformer model (`transformer_model.pth`) is tracked using **Git LFS**. 

### Prerequisites
Ensure Git LFS is installed on your system:
```bash
git lfs install
```

The model will be automatically downloaded when you clone the repository.

---

## 📚 Notebook

The `Hinglish_to_English_Transformer.ipynb` notebook contains:

- 📝 Data preprocessing pipelines
- 🏋️ Model training and evaluation procedures  
- 💡 Translation examples with Hinglish sentences
- 📈 Performance metrics and visualizations

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Rushild25/Neural-Machine-Translation-Hinglish-English-Transformer.git
cd Neural-Machine-Translation-Hinglish-English-Transformer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Required Dependencies
- `torch` - PyTorch framework
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `tqdm` - Progress bars

---

## 🚀 Usage

### Option 1: Using Jupyter Notebook
```bash
jupyter notebook Hinglish_to_English_Transformer.ipynb
```

### Option 2: Using Web Application
```bash
streamlit run app.py
```

### Option 3: Using Python Script
```bash
python translate_hinglish.py
```

### Option 4: Programmatic Usage
```python
from translate import Translator

# Initialize translator with pre-trained model
translator = Translator(model_path="transformer_model.pth")

# Translate Hinglish text to English
hinglish_text = "Mujhe chai chahiye"
english_text = translator.translate(hinglish_text)
print(english_text)  # Output: "I want tea"
```

---

## 📁 Project Structure

```
Neural-Machine-Translation-Hinglish-English-Transformer/
│
├── 📂 data/                                    # Dataset files (optional)
├── 🧠 transformer_model.pth                   # Pre-trained model (Git LFS)
├── 📓 Hinglish_to_English_Transformer.ipynb   # Main training notebook
├── 🚀 app.py                                  # Streamlit/Flask web application
├── ⚙️ model_components.py                     # Transformer model architecture
└── 📖 README.md                               # Project documentation
```

---

## 📝 Important Notes

- 🗂️ **Large Files**: Model files are tracked with Git LFS for efficient version control
- 🌐 **External Dataset**: Training data is hosted externally due to GitHub size constraints  
- ⚙️ **Preprocessing**: Follow preprocessing steps carefully when using custom datasets
- 💾 **Storage**: Ensure adequate disk space for model and dataset files

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Contact

**Developed by Rushil Dhingra**

- 🐙 GitHub: [@Rushild25](https://github.com/Rushild25)
- 🔗 Repository: [Neural-Machine-Translation-Hinglish-English-Transformer](https://github.com/Rushild25/Neural-Machine-Translation-Hinglish-English-Transformer)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

*Made with ❤️ for the open-source community*

</div>
