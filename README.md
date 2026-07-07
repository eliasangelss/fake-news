# 📰 Fake News Classification

A dual-classification BERT model that determines both the authenticity and the topic of a news article.

---

## Overview

In 2025, I participated in MIT FutureMakers, a six-week program in collaboration with SureStart to teach students about data analysis, mobile app development, and deep learning. Our team completed three practice neural networks before creating a final capstone project; this was the third of the three networks that we created. The Fake News dataset is another popular set for those learning to get started with MLP's and Neural Nets, but the primary goal of this project was to learn supervised learning with text data and how to utilize the BERT model in the process. The network learns to categorize the news articles into two different categories: the validity of the article (real or fake) and the topic of the article (world news or politics), Cells were set aside for Optuna and ExAI methods to see where the model was coming up with inconsistencies, and the accuracy was determined via graphs and reports.

---

## Dataset

- **Source:** [Kaggle Fake News Detection Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
- **Size:** 44,898 articles, 21,471 real sources and 23,481 fake sources
- **Features:** Title, text, subject, date
- **Target classes:** *Real news*, *Fake news*, *Politics news*, *World news*

---

## Results

With a 400-sample test set, the validity classifer achieved 98.50% accuracy while the topic classifier reached 92%—displaying the contrast between grasping writing style and grasping subject matter.

### Confusion Matrix (Fake/Real)

<img src="images/news_heatmap_a.png" width="500"/>

### Classification Report (Fake/Real)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0.0 (Fake) | 0.9883 | 0.9769 | 0.9826 | 173 |
| 0.1 (Real) | 0.9912 | 0.9912 | 0.9912 | 227 |
| Accuracy |  |  | 0.9850 | 400 |
| Macro Avg | 0.9854 | 0.9840 | 0.9847 | 400 | 
| Weighted Avg | 0.9850 | 0.9850 | 0.9850 | 400 |

### Confusion Matrix (Politics/World)

<img src="images/news_heatmap_b.png" width="500"/>

### Classification Report (Politics/World)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0.0 (Politics) | 0.9382 | 0.8883 | 0.9126 | 188 |
| 0.1 (World) | 0.9054 | 0.9481 | 0.9263 | 212 |
| Accuracy |  |  | 0.92 | 400 |
| Macro Avg | 0.9218 | 0.9182 | 0.9194 | 400 | 
| Weighted Avg | 0.9208 | 0.9200 | 0.9198 | 400 |

---

## Key Takeaways

- **Weighted classes** were used for the authenticity classifier due to the imbalance after removing duplicated articles; by adjusting the loss calculations, the penalty for getting the minority group incorrect (fake news) was much greater.
   - Class weighting wasn't implemented for the topic classifier, which (other than the natural overlap between politics and world news) may be part of the reason that it wasn't as accurate.
- When searching for the best hyperparameters, **Optuna** decided that the ideal learning rate was 3.46e-5 and the ideal batch size was 4, determining how many samples the model views before updating.
   - The average F1 score increased from 0.8963 (across both tasks) to 0.9900 and 0.9175 after implementation.
- Explainable AI methods (such as **LIME**), which help to explain the results produced by algorithms found that the term "reuters" was the greatest indicator of whether or not an article was real or fake at +0.5075.
   - I went back through and removed any reuters tokens in my second iteration, and found that the accuracy shift from removing the leakage was similar to the impact of normal run-to-run parameter adjustments (in other words, it's closer to noise than indicating a meaningful change.)
   - That being said, the minimal change showed me that the model wasn't relying too heavily on shortcuts; rather, the reasoning behind each choice is more likely to be contextual. It also revealed other minor artifacts (a byline and HTML fragment) that could be removed in a future iteration.
 
<br>
<img src="images/news_tokens.png" width="500"/>
<sub><i> Token frequency distribution within the dataset </i></sub>

---

## Setup

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Notebook

```bash
git clone https://github.com/eliasangelss/fake-news.git
cd fake-news
jupyter notebook model.ipynb
```

---

## Project Structure

```
fake-news/
├── model.ipynb       # Includes analysis, modeling, evaluation, and my notes
├── newsData          # Folder containing the dataset
├── requirements.txt
└── README.md
```

---

## Model Architecture

```
Input (tokenized text → input_ids + attention_mask)
  → BertModel (bert-base-uncased)
  → [CLS] token embedding (768-dim)
  → Linear(768 → 1) Authenticity Head
  → Linear(768 → 1) Subject Head
```

- **Train-Test Split:** 1600 train / 400 test
- **Optimizer:** AdamW (lr = 3.46e-5)
- **Loss:** BCEWithLogitsLoss, pos_weight applied to authenticity head 
- **Regularization:** Early stopping (patience = 2 epochs, batch_size = 4)

---

## Tools & Libraries

- Python, Jupyter Notebook, IPython
- PyTorch, Transformers, deep_translator
- scikit-learn, pandas, NumPy
- matplotlib, seaborn
- Optuna, LIME

---

## License

MIT
