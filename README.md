# 📩 Spam Classifier with Model Comparison (RNN, LSTM, GRU)

This project is a **Spam Message Classifier** built using **deep learning models** (RNN, LSTM, and GRU).
It allows users to compare the performance of different models, analyze evaluation metrics, and test predictions on both **test datasets** and **raw input text**.
A **Streamlit frontend** is also provided for interactive use.

---

## 🚀 Features

* Preprocesses raw text (lowercasing, stopword removal, stemming, tokenization).
* Implements and trains **RNN, LSTM, and GRU models**.
* Performs **hyperparameter tuning** to find the best thresholds.
* Evaluates models using **accuracy, confusion matrix, ROC-AUC, Precision-Recall**.
* Provides predictions for:

  * Test dataset
  * Raw user input (custom text)
* Interactive **Streamlit UI** for easy usage.
     link --> https://spam-classifier-model-comparison-eotpahqdpniqmgathvlnge.streamlit.app/

---

## 📊 Model Performance

### 🔹 Base Models (without tuning)

* **RNN** → Accuracy: `0.5636`
* **LSTM** → Accuracy: `0.6756`
* **GRU** → Accuracy: `0.9866`

### 🔹 After Hyperparameter Tuning

* **RNN** → Accuracy: `0.6503`
* **LSTM** → Accuracy: `0.9892`
* **GRU** → Accuracy: `0.9884`

### 🔹 Final Test Accuracy

* **RNN** → `0.6136`
* **LSTM** → `0.9887`
* **GRU** → `0.9881`

✅ Clearly, **LSTM and GRU** outperform RNN by a large margin.

---

## 🔧 Data Preprocessing

Text preprocessing is done with **NLTK** and includes:

1. Lowercasing
2. Tokenization
3. Removing stopwords and punctuation
4. Keeping only alphanumeric tokens
5. Applying **Porter Stemming**
6. Joining back into cleaned text

👉 Example transformation:

```
Original: "Congratulations!!! You won a FREE iPhone, click here to claim."
Transformed: "congratul won free iphon click claim"
```

---

## 🏗️ Model Training

* Used **Keras (TensorFlow)** for model building.
* **Embedding Layer + Simple RNN/LSTM/GRU + Dense Layer**.
* Loss: `binary_crossentropy`
* Optimizer: `adam`
* Metrics: `accuracy`

---

## ⚡ Hyperparameter Tuning

* Thresholds for classification were tuned using **ROC-AUC & Precision-Recall trade-off**.
* Best thresholds:

  * **RNN:** `0.3687`
  * **LSTM:** `0.3130`
  * **GRU:** `0.7641`

---

## 📈 Evaluation

Metrics used:

* **Confusion Matrix**
* **Classification Report (Precision, Recall, F1-score)**
* **ROC Curve & AUC Score**
* **Precision-Recall Curve**

Both **test dataset** predictions and **raw user input** predictions were compared across models.

---

## 🖥️ Streamlit Frontend

The app provides:

* Text input box ✍️
* Model selector (RNN, LSTM, GRU) 🧠
* Prediction result:

  * Spam (0) 🚨
  * Ham (1) ✅
* Shows probability and threshold used.

👉 Example Prediction:

```
Message: "Congratulations! You won a free iPhone."
Model: GRU
Prediction: 🚨 Spam (0)
Probability: 0.9812
Threshold: 0.764
```
---

## 🔮 Future Improvements

* Add **BiLSTM and Transformer-based models (BERT, DistilBERT)**.
* Improve preprocessing with **lemmatization**.
* Deploy to **Hugging Face Spaces / Streamlit Cloud**.

---

## ✨ Results

* **GRU and LSTM are highly effective**, achieving \~99% accuracy.
* RNN performs poorly compared to GRU/LSTM.
* Final deployed app allows **real-time spam classification** with model comparison.



