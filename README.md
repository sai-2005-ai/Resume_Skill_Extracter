# 🧠 Resume Skill Extraction and Classification using DistilBERT

This project presents a powerful NLP-based system for automated resume screening and skill extraction using the DistilBERT transformer model. It enables accurate classification of resumes into job roles based on deep contextual understanding, making it ideal for recruitment automation.

---

## ✨ Key Features

- 📄 **Resume Parsing**: Handles unstructured text from resumes (.pdf, .docx)
- 🧹 **Preprocessing**: Tokenization, stopword removal, lowercasing, punctuation stripping
- 🧠 **Embedding**: Uses pre-trained **DistilBERT** to capture context and semantics
- 📊 **Classification**: Predicts job roles using a softmax layer on top of DistilBERT embeddings
- 📈 **Evaluation**: Calculates Accuracy, Precision, Recall, F1-Score, and Loss
- 📉 **Visualization**: Plots confusion matrix and training/validation curves

---

## 📁 Dataset

- 📦 900 real-world resumes labeled across multiple job categories:
  - Software Developer, Data Analyst, Web Developer, CRM Analyst, etc.
- Formats: `.pdf`, `.docx`, `.txt`
- Splits:
  - 80% Training
  - 20% Testing

---

## 📈 Model Performance

| Metric     | Score (%) |
|------------|-----------|
| Accuracy   | 93.2      |
| Precision  | 92.4      |
| Recall     | 91.1      |
| F1-Score   | 91.7      |
| Loss       | 0.17      |

> ✅ Outperforms models like Naive Bayes, SVM, LSTM, and even BERT-base — with faster performance and lighter compute using DistilBERT.

---

## 🛠 Tech Stack

- **Python 3**
- **NLP**: Hugging Face `transformers` (DistilBERT)
- **ML Framework**: PyTorch
- **Evaluation**: Scikit-learn
- **Data Handling**: pandas, NumPy
- **Visualization**: matplotlib

---

## 🚀 How to Run

1. **Install dependencies**:
   ```bash
   pip install torch transformers scikit-learn pandas matplotlib

## Run the Notebook:
Open resume_skill_extraction_nlp.ipynb in Jupyter or VS Code and follow the cells step-by-step.

## Upload Resumes:
Add your resume data in supported formats for parsing and classification.

## Future Enhancements

    🗣️ Multilingual resume support

    🏷️ Multi-label classification (for multi-role candidates)

    🧠 NER-based skill extraction (e.g., using SpaCy's SkillNER)

    ⚙️ API deployment for enterprise integration

  ## Author

    Batta Sai Sailu
    Final Year CSE Student, [VIT-AP]
    ✉️ GitHub: sai-2005-ai

  ##  License

This project is for academic and educational use. You're free to use or modify it with proper attribution.
## References

    Sanh et al., DistilBERT: A distilled version of BERT – arXiv:1910.01108

    Hugging Face Transformers – https://huggingface.co

    Scikit-learn Metrics – https://scikit-learn.org
