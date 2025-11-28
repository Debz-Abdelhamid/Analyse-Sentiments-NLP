# NLP Emotion Classification with DistilBERT

This project demonstrates emotion classification using DistilBERT, a lightweight transformer model. It fine-tunes a pre-trained DistilBERT model on the HuggingFace `emotion` dataset to classify text into six emotion categories: sadness, joy, love, anger, fear, and surprise. The trained model is then applied to analyze customer reviews from a CSV file.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Results](#results)
- [License](#license)

## Description
The notebook (`NLP_Emotion_Classification_Notebook.ipynb`) walks through the complete workflow:
- Loading and exploring the emotion dataset
- Preprocessing and tokenizing text using DistilBERT tokenizer
- Fine-tuning the model for sequence classification
- Evaluating performance on test data
- Testing on custom sentences
- Analyzing customer reviews from `customer_reviews.csv` and saving predictions to `customer_reviews_with_predictions.csv`

## Installation
Ensure you have Python 3.7+ installed. Install the required libraries using pip:

```bash
pip install transformers datasets accelerate torch scikit-learn matplotlib seaborn pandas tqdm
```

For GPU support, ensure PyTorch is installed with CUDA if available.

## Usage
1. Place your `customer_reviews.csv` file in the same directory as the notebook. The CSV should have a column named `review` containing the text to analyze.
2. Open and run the notebook `NLP_Emotion_Classification_Notebook.ipynb` in Jupyter or Google Colab.
3. Follow the cells sequentially to train the model and generate predictions.
4. The output will include:
   - Trained model saved in `distilbert-emotion/`
   - Predictions for customer reviews in `customer_reviews_with_predictions.csv`

## Dataset
- **Training Data**: HuggingFace `dair-ai/emotion` dataset
  - 6 emotion labels: sadness, joy, love, anger, fear, surprise
  - Used for fine-tuning the DistilBERT model
- **Customer Reviews**: Custom CSV file (`customer_reviews.csv`)
  - Column: `review` (text data)
  - Predictions include label, probabilities for each emotion, and saved to `customer_reviews_with_predictions.csv`

## Model Details
- **Model**: DistilBERT (distilbert-base-uncased)
- **Task**: Sequence Classification
- **Training**:
  - Epochs: 2
  - Batch Size: 64
  - Learning Rate: 2e-5
  - Optimizer: AdamW with weight decay
- **Metrics**: Accuracy, Weighted F1-Score, Confusion Matrix

## Results
- **Test Set Performance** (example from notebook):
  - Accuracy: ~0.92
  - Weighted F1: ~0.92
- The model successfully classifies emotions in custom text and customer reviews.
- Visualizations include label distributions and confusion matrices.

## License
This project is for educational purposes. Please refer to the licenses of the datasets and models used (HuggingFace, DistilBERT).
