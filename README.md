# Spam Email/SMS Classifier

## 1. Objective
The goal of this project is to **classify messages as "spam" or "ham" (not spam)**.  

**Motivation:** Spam messages are unwanted and sometimes malicious. Automatically detecting them improves communication safety.  

**Challenge:** The dataset is **imbalanced** (more ham than spam), so special handling is required.

---

## 2. Dataset
- **Source:** [SMS Spam Collection Dataset (UCI / Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
- **Size:** 5,572 messages  
- **Columns:**  
  - `label`: "ham" or "spam"  
  - `message`: raw SMS text  

**Label Distribution:**

| Label | Count | Proportion |
|-------|-------|-----------|
| ham   | 4825  | 86.6%     |
| spam  | 747   | 13.4%     |

**Example Messages:**

- Ham: `"Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."`
- Spam: `"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121..."`

---

## 3. Exploratory Data Analysis (EDA)
- Checked for **null values** → none found.  
- Added **message length** feature:  
  - Average ham length ≈ 71  
  - Average spam length ≈ 139  
- Visualized **word clouds** and word frequency counts for spam vs ham  

**Insights:**  
- Spam messages are generally longer and contain words like `"free"`, `"win"`, `"urgent"`.  
- Ham messages are more conversational.

---

## 4. Data Preprocessing
- **Label encoding:** `"ham" → 0`, `"spam" → 1`  
- **Text cleaning:**  
  - Lowercased messages  
  - Removed punctuation  
  - Removed stopwords (for vectorization later)  
- Column created: `clean_message`

---

## 5. Feature Extraction
- Used **TF-IDF vectorization** to convert text into numerical features  
- Explored **unigrams** and **bigrams**; **unigrams + SVM** worked best for accuracy and F1

---

## 6. Train/Test Split
- Dataset split into **training (80%)** and **testing (20%)** subsets  
- Used **stratified split** to preserve spam vs ham ratio

---

## 7. Modeling
- **Initial models tried:**  
  - Multinomial Naive Bayes → baseline  
  - Logistic Regression → slightly better  
  - SVM + TF-IDF → best accuracy and F1  

**Observation:**  
- Standard SVM + TF-IDF:  
  - Accuracy ≈ 0.983  
  - Spam recall ≈ 0.88 (some spam missed)

---

## 8. Handling Class Imbalance
- Spam messages are rarer → used **class weighting in SVM**  
- Results:  
  - Spam recall ↑ from 0.88 → 0.92  
  - Accuracy ≈ 0.986

---

## 9. Threshold Tuning
- Default threshold = 0.5  
- Converted decision function scores to pseudo-probabilities and explored thresholds  

**Threshold selection:**

| Threshold | Spam Recall | Spam Precision | F1 (spam) |
|-----------|------------|----------------|-----------|
| 0.41      | 0.95       | 0.88           | 0.91      |
| 0.48–0.53 | 0.92–0.93  | 0.96–0.99      | 0.94–0.95 |

- **Chosen threshold:** 0.41 to prioritize spam recall

---

## 10. Final Pipeline & Prediction
- Combined **TF-IDF vectorizer + SVM classifier + custom threshold** into a pipeline  
- Tested on custom messages, e.g., iPhone giveaway and class reminders  
- Predictions include **spam probability** to quantify confidence

**Sample predictions:**

| Message | Prediction | Spam Probability |
|---------|-----------|----------------|
| Congratulations! You’ve been selected for a free iPhone giveaway. Click here! | spam | 0.4896 |
| Are you coming to class today? | ham | 0.2485 |
| URGENT! Your account will be suspended unless you verify now. | spam | 0.5056 |
| Get rich quick with this amazing investment opportunity!!! | ham | 0.3644 |

---

## 11. Evaluation Metrics (Test Set)
- Accuracy: 0.986  
- Spam recall: 0.92  
- Spam precision: 0.97  

**Confusion Matrix:**

| Actual\Pred | Ham | Spam |
|------------|-----|------|
| Ham        | 962 | 4    |
| Spam       | 12  | 137  |

---

## 12. Saved Files for Deployment
- `svm_spam_pipeline.pkl` → Full trained pipeline (TF-IDF + SVM)  
- `spam_threshold.pkl` → Custom threshold for spam detection (0.41)

---

## 13. Conclusion & Next Steps
- Pipeline detects spam with **high recall and precision**  
- Threshold tuning allows flexibility for prioritizing spam detection  
- Future improvements:  
  - Test on larger batch of messages  
  - Ensemble models for better performance  
  - Integrate into real-world SMS/email filtering systems
