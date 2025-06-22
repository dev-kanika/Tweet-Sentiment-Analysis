# Tweet Sentiment Analysis using Deep Learning & Transfer Learning

**Course Project | NLP | Loyalist College**

This project focuses on classifying tweets into **positive**, **negative**, or **neutral** sentiments using multiple deep learning architectures and traditional machine learning models. It leverages pre-trained embeddings (GloVe, word2vec), performs empirical tuning, and uses explainability tools like SHAP and LIME to interpret predictions.

## Overview
This project was developed as part of an academic assignment in NLP. We explored a multi-model pipeline where five different architectures were built, trained, and tuned using a custom preprocessed tweet dataset.

### Pipeline Structure
1. **Preprocessing**: Cleaned tweets (punctuation, emoji, case-folding)
2. **Feature Engineering**: `tweet_length`, `lexicon_score`
3. **Embedding**: GloVe, word2vec, TF-IDF
4. **Modeling**:
   - Unidirectional RNN with TF-IDF
   - Unidirectional LSTM with word2vec
   - Bidirectional GRU with learned embedding
   - Bidirectional LSTM with learned embedding
   - SVM with GloVe and TF-IDF
5. **Evaluation**: Accuracy, F1-score, AUC, Confusion Matrix
6. **Explainability**: SHAP, LIME

## Models Implemented
- **RNN + TF-IDF**
- **LSTM + word2vec** (64 to 300 vector sizes, CBOW/Skipgram)
- **BiGRU** (learned embedding)
- **BiLSTM** (learned embedding)
- **SVM + GloVe/TF-IDF**

## Results Summary
| Model                          | Accuracy | F1 (Positive) | Macro AUC |
|-------------------------------|----------|---------------|-----------|
| SVM (TF-IDF)                  | 56%      | 0.59          | 0.64      |
| LSTM (word2vec)               | 52%      | 0.58          | 0.69      |
| BiGRU (learned embedding)     | 53%      | 0.60          | 0.66      |
| BiLSTM (learned embedding)    | 54%      | 0.61          | 0.68      |
| RNN (TF-IDF)                  | 51%      | 0.56          | 0.62      |

## Tech Stack
- **Languages**: Python
- **Libraries**: TensorFlow, Keras, scikit-learn, Gensim, SHAP, LIME
- **Embeddings**: GloVe, word2vec, TF-IDF
- **Tools**: Google Colab, GitHub, Matplotlib, Seaborn

## Project Structure
```
├── data/
│   ├── processed/ (train/test/val CSVs)
│   └── raw/ (Sentiment_Data.csv)
├── model-tuning/
│   └── model/ (saved .h5 and .pkl files, encoders, tokenizers)
├── notebooks/
│   └── Assignment-2.ipynb
```

## Interpretability
To make model predictions transparent, we used:
- **LIME**: Local explanations for individual predictions
- **SHAP**: Global and local feature importance visualizations

These tools helped debug model behavior and identify key drivers behind sentiment classification.

## Future Improvements
- Integrate BERT-based models (e.g., DistilBERT, RoBERTa)
- Add multilingual support for tweets
- Fine-tune embeddings for domain-specific sentiment

## Authors & Acknowledgements
Developed by Group of students as part of the NLP course assignment at Loyalist College.

*Feel free to fork, improve, or explore!*
