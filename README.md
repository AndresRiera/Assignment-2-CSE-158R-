# Assignment-2-CSE-158R

Andres Riera Ortiz Parraga & Rosey Gutierrez Meza

## Contents

 - `notebook.ipynb` - Jupyter Notebook with EDA Analysis, Model implementations and evaluations.

### Dataset: Amazon Reviews 2023 — Electronics.jsonl
Link : `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz`

## Part 1 — Predictive task, evaluation, baselines, and validity
### 1.1 Predictive task

Primary task: Predict the star rating (1–5) that a reviewer gives a product in the Electronics category.

- Type: Regression (predict numeric rating) with optional discrete rounding to 1..5 for classification-like evaluation.

Why this task?

- It is directly useful (e.g., product recommender systems, review summarization, product quality modeling).

- It lets us combine textual signals (review text), metadata (verified_purchase, review length), and collaborative signals (user/item historical averages or latent factors).

### 1.2 How will the model be evaluated

Primary evaluation metrics on held-out test data:

- MSE (Mean Squared Error) and RMSE — favors penalizing larger errors.

- MAE (Mean Absolute Error) — interpretable in rating units.

- If we round to discrete ratings, also report classification accuracy and confusion matrix.

Validation procedure:

- Hold-out split: Reserve 5–10% of the earliest/largest dataset chunk as a test set, keep a validation set (e.g., last 10k interactions) to tune hyperparameters. Use the same splitting strategy as the course (first 190k train, last 10k validation) if required.

- Perform k-fold cross-validation (k=3) for smaller-scale experiments to check stability.

Important secondary checks:

- Evaluate per-item and per-user RMSE to detect bias where rare users/items produce high errors.

- Report distribution of residuals (histograms) and plot predicted vs actual ratings.

### 1.3 Baselines for comparison

Start with simple baselines:

1. Global mean: Predict the global average rating.

2. User mean / item mean: If user or item is seen in training. 

3. Bias-only (alpha + bu + bi): Model with global mean + regularized user and item biases solved via SGD or closed-form updates.

4. Similarity-based (item-item using Jaccard or cosine): Weighted average of ratings on similar items.

5. Text-only model: TF-IDF of review text + linear regressor (or logistic regression for discrete labels).

6. Latent factor model: Matrix factorization (bias + latent factors) trained with SGD.

### 1.4 Assessing the validity of predictions

- Calibration: Check whether predicted continuous outputs match empirical distribution (apply isotonic regression if necessary).

- Residual analysis: Plot residuals vs predicted and vs features (item popularity, review length, verified flag).

- Stratified performance: Report RMSE by rating bins, by item popularity deciles, and by user activity to ensure fairness/stability.

- Held-out / private set: Avoid overfitting by using internal cross-validation.

- Error analysis: Manually inspect examples with large absolute error to find missing signals.

## Part 2 — Exploratory analysis, data collection, preprocessing, code, and discussion

### 2.1 Context and Data Source

The dataset comes from the **Amazon Reviews 2023 - Electronics** category, part of a large-scale product review corpus widely used in academic research for studying recommender systems, sentiment analysis, and review text mining. The data is sourced from authentic Amazon user reviews spanning multiple years.

**Data Source:** https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz

**Dataset Characteristics:**
- **Format:** JSONL (JSON Lines) - one review per line, gzip compressed
- **Subset Used:** First 250,000 reviews (randomly sampled from full dataset)
- **Original Size:** ~20GB compressed, millions of reviews
- **Collection Period:** Reviews span from early 2000s through 2023

**Data Schema:**
Each review contains the following fields:
- `rating`: Star rating (1-5, float)
- `text`: Review text content (string)
- `title`: Review title/summary (string)
- `asin`: Amazon Standard Identification Number - unique product ID
- `parent_asin`: Parent product identifier (for product variants)
- `user_id`: Anonymized reviewer identifier
- `timestamp`: Unix timestamp in milliseconds
- `verified_purchase`: Boolean flag indicating if purchase was verified by Amazon
- `helpful_vote`: Number of users who found the review helpful
- `images`: List of user-uploaded image URLs (if any)

**Why This Dataset?**

This dataset is ideal for our rating prediction task because:
1. **Large scale:** 250k reviews provide sufficient data for both collaborative and content-based models
2. **Rich features:** Combines collaborative signals (user/item IDs), content (review text), and metadata (verified purchase, helpfulness)
3. **Real-world relevance:** Actual e-commerce data reflects genuine user behavior and rating patterns
4. **Temporal ordering:** Timestamps enable realistic train/test splits simulating production deployment
5. **Quality diversity:** Electronics category includes diverse products (from cables to cameras) with varying review quality

### 2.2 Data Preprocessing and Feature Engineering

**Initial Data Cleaning:**
```python
# Load subset of 250,000 reviews
df = pd.read_json('Electronics.jsonl', lines=True, nrows=250000)

# Remove reviews with missing critical fields
df = df[df['rating'].notna() & df['text'].notna()].copy()
```

**Results:**
- Initial size: 250,000 reviews
- After cleaning: 250,000 reviews (no missing values in critical fields)
- Data quality is high with minimal null values

**Temporal Processing:**
```python
# Convert Unix timestamp (milliseconds) to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')

# Extract temporal features
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['quarter'] = df['timestamp'].dt.quarter
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['season'] = df['month'].apply(lambda x: 
    'Winter' if x in [12, 1, 2] else
    'Spring' if x in [3, 4, 5] else
    'Summer' if x in [6, 7, 8] else 'Fall')
```

**Date Range:** Reviews span from 2000 to 2023 (23 years of data)

**Feature Engineering:**

We created several derived features to capture review characteristics:

1. **Text Length Features:**
```python
# Word count (primary length metric)
df['review_len_words'] = df['text'].str.split().str.len()

# Character count
df['review_len_chars'] = df['text'].str.len()
```

2. **Text Sentiment/Style Features:**
```python
# Punctuation and emphasis indicators
df['exclamation_count'] = df['text'].str.count('!')
df['question_count'] = df['text'].str.count(r'\?')
df['all_caps_words'] = df['text'].apply(
    lambda x: len(re.findall(r'\b[A-Z]{2,}\b', str(x)))
)
```

These features capture emotional intensity and writing style, which correlate with rating extremes.

3. **Text Preprocessing for NLP Models:**
```python
def clean_text(text):
    """Remove HTML tags, URLs, and special characters"""
    text = re.sub(r']+>', ' ', text)  # Remove HTML tags like 
    text = re.sub(r'&[a-z]+;', ' ', text)  # Remove HTML entities
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Keep only letters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text.lower()

# Apply cleaning for text analysis
df['text_clean'] = df['text'].apply(clean_text)
```

4. **TF-IDF Vectorization (for modeling):**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=5000,        # Limit vocabulary size
    min_df=5,                 # Ignore ultra-rare words (< 5 docs)
    max_df=0.7,               # Ignore very common words (> 70% docs)
    stop_words='english',     # Remove common English stopwords
    ngram_range=(1, 2),       # Include unigrams and bigrams
    lowercase=True,
    strip_accents='unicode'
)
```

This creates a 5,000-dimensional sparse feature matrix capturing word and phrase frequencies.

**Data Quality Assessment:**

| Field | Non-Null Count | Null % | Notes |
|-------|---------------|--------|-------|
| rating | 250,000 | 0% | Core target variable |
| text | 250,000 | 0% | Primary content feature |
| user_id | 250,000 | 0% | Essential for collaborative filtering |
| asin | 250,000 | 0% | Essential for collaborative filtering |
| timestamp | 250,000 | 0% | Used for temporal splits |
| verified_purchase | 250,000 | 0% | Useful metadata |
| helpful_vote | 250,000 | 0% | Available but not used in this study |
| title | 248,234 | 0.7% | Minor missing values, not critical |
| images | Variable | - | Sparse, not used |

**Final Dataset Statistics:**
- **Total reviews:** 250,000
- **Unique users:** 87,432
- **Unique products (ASINs):** 45,891
- **Date range:** 2000-01-15 to 2023-11-28
- **Average review length:** 68 words (median: 43 words)
- **Sparsity:** 99.93% (only 0.07% of user-item pairs have reviews)

### 2.3 Exploratory Data Analysis: Key Findings

#### **Finding 1: Strong Positive Rating Skew**

**Rating Distribution:**
- 5-star: 65.3% (163,244 reviews)
- 4-star: 15.1% (37,628 reviews)
- 3-star: 7.1% (17,681 reviews)
- 2-star: 4.5% (11,321 reviews)
- 1-star: 8.0% (20,126 reviews)

**Mean rating:** 4.26 stars (significantly above midpoint of 3.0)

**Implications:**
- Simple "predict 5 stars for everything" baseline achieves 65% accuracy
- RMSE and MAE are more appropriate metrics than accuracy
- Models must avoid simply predicting the mean/mode
- Class imbalance suggests stratified evaluation is important
- Negative reviews (1-2 stars) are relatively rare but informative

**Visualization:** The rating distribution shows a clear J-shaped curve with massive concentration at 5 stars, moderate 4-star presence, and sparse mid-range ratings.

#### **Finding 2: Review Length Correlates Negatively with Rating**

**Average Word Count by Rating:**
- 1-star: 71.3 words
- 2-star: 87.4 words  
- 3-star: 92.1 words
- 4-star: 95.7 words
- 5-star: 56.2 words

**Key Observations:**
- **Negative reviews are longer:** Users write more when dissatisfied, often explaining what went wrong
- **Positive reviews are shorter:** Simple expressions like "Great product!" or "Works perfectly" are common
- **Mid-range reviews longest:** 3-4 star reviews contain nuanced explanations of pros/cons
- **High variance:** Box plots show extreme outliers up to 30,000+ characters (likely copy-pasted specs or detailed narratives)

**Implications for Modeling:**
- Review length is a weak predictive feature (negative correlation with rating)
- Text content matters more than length
- Very short reviews may be less informative for text-based models

**Distribution Shape:** Highly right-skewed with long tail; most reviews under 200 words, but some exceed 1000+ words

#### **Finding 3: Verified Purchases Show Distinct Rating Patterns**

**Verification Rate:** 82.4% of reviews are verified purchases

**Rating Distribution Comparison:**

| Rating | Verified % | Non-Verified % | Difference |
|--------|-----------|----------------|------------|
| 5-star | 64.2% | 69.8% | -5.6% |
| 4-star | 15.4% | 13.8% | +1.6% |
| 3-star | 7.3% | 6.1% | +1.2% |
| 2-star | 4.6% | 4.0% | +0.6% |
| 1-star | 8.5% | 6.3% | +2.2% |

**Average Rating:**
- Verified: 4.24 stars
- Non-verified: 4.32 stars
- **Difference: -0.08 stars**

**Key Observations:**
- Non-verified reviews skew slightly more positive
- Verified purchases have more balanced distribution (relatively more 1-stars and 4-stars)
- Possible explanation: Non-verified reviews may include promotional content or fake reviews

**Review Length Comparison:**
- Verified: 67.2 words (median: 42)
- Non-verified: 71.8 words (median: 48)
- Non-verified reviews are slightly longer

**Implications:**
- `verified_purchase` is a useful feature for models
- Verified reviews may be more trustworthy/representative
- Including verification status could improve prediction accuracy

#### **Finding 4: Temporal Trends Reveal Rating Inflation Over Time**

**Average Rating by Year:**
- 2000-2005: 3.95 stars
- 2006-2010: 4.12 stars
- 2011-2015: 4.23 stars
- 2016-2020: 4.31 stars
- 2021-2023: 4.38 stars

**Observation:** Steady upward trend of ~0.1 stars per 5-year period

**Possible Explanations:**
1. **Selection bias:** Early adopters more critical; mass market more satisfied
2. **Platform changes:** Amazon encouraged positive reviews through follow-up emails
3. **Product quality improvement:** Electronics genuinely improved over 20+ years
4. **Review inflation:** Social norms shifted toward higher ratings
5. **Survivorship bias:** Poorly-rated products discontinued, data only shows survivors

**Review Volume Trends:**
- Exponential growth from 2000 (few hundred/year) to 2020 (tens of thousands/year)
- Sharp increase 2015-2020 coinciding with Amazon's growth
- Slight decline 2021-2023 (possible COVID-19 supply chain effects or data collection end)

**Seasonal Patterns:**
- **Highest review volume:** November-December (holiday shopping)
- **Lowest review volume:** January-February (post-holiday lull)
- **Rating differences by season:** Negligible (±0.02 stars)
- **Day of week:** No significant pattern (online reviews 24/7)

**Implications for Modeling:**
- Temporal train/test split is essential (avoid data leakage from future)
- Models must generalize across time periods
- Test set (2023 reviews) may have higher baseline ratings than training set (2000-2022)

#### **Finding 5: Extreme Power Law Distribution in User and Item Activity**

**User Review Counts:**
- **Median:** 1 review per user (50% of users wrote only 1 review)
- **Mean:** 2.86 reviews per user
- **Top 1% users:** Account for 23.7% of all reviews
- **Top 10 users:** 1,034, 820, 764... reviews (power users)
- **Distribution:** Highly right-skewed power law

**User Types:**
- One-time reviewers: 31.6% of users (27,645 users)
- Light reviewers (2-5): 43.6% (38,119 users)
- Medium reviewers (6-20): 20.6% (18,011 users)  
- Heavy reviewers (20+): 4.2% (3,657 users)

**Item Review Counts:**
- **Median:** 2 reviews per product
- **Mean:** 5.45 reviews per product
- **Top 1% products:** Account for 31.2% of all reviews
- **Most-reviewed products:** 4,913, 3,876, 2,234... reviews (popular items)
- **Distribution:** Power law, even more extreme than users

**Product Types:**
- Rare products (≤1 review): 18.2% of products (8,352 items)
- Uncommon (2-10 reviews): 58.3% (26,754 items)
- Popular (11-100 reviews): 20.1% (9,224 items)
- Very popular (100+): 3.4% (1,561 items)

**Top-Reviewed Products:**
The most-reviewed items tend to be:
- Universal accessories (cables, chargers, screen protectors)
- Popular consumer electronics (Amazon Echo, Fire TV Stick)
- High-variance products (cheap items with inconsistent quality)

**Average Rating by Popularity:**
- Rare products (1 review): 4.31 stars
- Uncommon (2-10): 4.24 stars
- Popular (11-100): 4.27 stars
- Very popular (100+): 4.19 stars

**Observation:** Very popular products rated slightly lower (survivor bias: bad products accumulate negative reviews before being discontinued)

**Implications for Modeling:**
- **Cold-start problem is severe:** 60%+ of test users/items may be new or rare
- Collaborative filtering will struggle on sparse data
- Content-based models (using text) may be more robust
- Need to evaluate stratified performance by user/item activity
- Matrix factorization requires careful regularization to avoid overfitting popular items

#### **Finding 6: Text Content Reveals Clear Sentiment Patterns**

**Word Cloud Analysis:**

**1-Star Reviews (Negative):**
- Top words: "not", "product", "time", "return", "money", "back", "waste"
- Common phrases: "didn't work", "stopped working", "waste of money", "poor quality"
- Emphasis: High use of ALL CAPS (1.2 words/review) and exclamation marks (0.38/review)

**3-Star Reviews (Mixed/Neutral):**
- Top words: "good", "okay", "works", "but", "price", "case", "sound"
- Common phrases: "works well but", "good for price", "decent but"
- Emphasis: Moderate CAPS (1.1 words/review) and questions (0.10/review)

**5-Star Reviews (Positive):**
- Top words: "great", "love", "perfect", "excellent", "highly", "recommend", "best"
- Common phrases: "highly recommend", "works great", "love this", "best purchase"
- Emphasis: Lower CAPS (0.98 words/review), high exclamation marks (0.43/review)

**Punctuation and Emphasis by Rating:**

| Rating | Avg ! Count | Avg ? Count | Avg CAPS Words |
|--------|-------------|-------------|----------------|
| 1-star | 0.38 | 0.10 | 1.20 |
| 2-star | 0.21 | 0.11 | 1.09 |
| 3-star | 0.17 | 0.10 | 1.01 |
| 4-star | 0.19 | 0.08 | 0.99 |
| 5-star | 0.43 | 0.03 | 0.68 |

**Observations:**
- **Exclamation marks:** U-shaped pattern (high in 1-star anger and 5-star excitement)
- **Question marks:** Slightly higher in negative reviews (confusion/frustration)
- **ALL CAPS:** More common in negative reviews (shouting/emphasis on problems)

**Implications for Modeling:**
- Text-based models should capture these sentiment signals
- TF-IDF will naturally weight discriminative words like "excellent" vs "terrible"
- Simple punctuation features provide weak predictive power
- N-grams (bigrams) capture phrases like "didn't work" better than unigrams alone

### 2.4 Summary and Implications for Modeling

Based on our EDA, we can draw the following conclusions for model development:

**Key Takeaways:**

1. **Strong class imbalance requires RMSE/MAE over accuracy**
   - 65% 5-star reviews make accuracy misleading
   - Must evaluate across all rating bins, not just overall

2. **Cold-start problem will be severe**
   - 50%+ users have ≤1 review in training
   - 45%+ products have ≤2 reviews in training
   - Text-based models will be critical for new users/items

3. **Text content is highly informative**
   - Clear sentiment patterns in word usage
   - Review length negatively correlates with rating
   - TF-IDF should capture discriminative vocabulary

4. **Temporal ordering is essential**
   - Rating inflation over time (4.0 → 4.4 stars)
   - Must use chronological train/test split
   - Test set may be slightly harder due to distribution shift

5. **Verified purchase is a useful signal**
   - 8% difference in rating distributions
   - Should be included as model feature

6. **Collaborative filtering will struggle**
   - Extreme sparsity (99.93%)
   - Power law distributions
   - Need strong regularization

**Recommended Modeling Strategy:**

Based on these findings, we will implement:
- **Baseline models:** Global mean, user/item means, bias model
- **Content-based:** TF-IDF + Ridge (should handle cold-start well)
- **Collaborative:** Matrix factorization (for warm-start comparison)
- **Hybrid:** Ensemble combining strengths of above

We expect text-based models to outperform pure collaborative filtering due to the severe cold-start problem and rich text content.

## Part 3 — Modeling

### 3.1 Context and Goal

Our goal is to predict star ratings (1-5) for Amazon Electronics reviews using a combination of collaborative filtering, content-based, and hybrid approaches. We formulate this as a regression problem where we predict continuous ratings, then optionally round to discrete values for classification metrics.

**Task Formulation:**
- **Input:** User ID, product ASIN, review text, and temporal information
- **Output:** Predicted rating (continuous: 1.0-5.0)
- **Optimization:** Minimize Mean Squared Error (MSE) on held-out test data
- **Evaluation:** RMSE (primary), MAE, and classification accuracy (secondary)

**Data Split Strategy:**
We use a temporal split to simulate real-world deployment:
- **Training:** First 190,000 reviews (earliest chronologically)
- **Validation:** Next 10,000 reviews (for hyperparameter tuning)
- **Test:** Remaining 50,000 reviews (most recent, unseen data)

This temporal ordering ensures our models are evaluated on future data they haven't seen, preventing data leakage and better simulating production conditions.

**Training Data Distribution:**
- 5-star: 124,087 (65.3%)
- 4-star: 28,681 (15.1%)
- 3-star: 13,537 (7.1%)
- 2-star: 8,584 (4.5%)
- 1-star: 15,111 (8.0%)

The strong class imbalance (positive skew) motivates our choice of RMSE/MAE over simple accuracy as primary metrics, and influences our modeling approach to avoid simply predicting the mean.

### 3.2 Discussing Model Advantages / Disadvantages, Challenges and Complexity

We implement five distinct modeling approaches, each with different strengths and computational complexity:

#### **Baseline Models:**

**1. Global Mean (Trivial Baseline)**
- **Approach:** Predict the training set mean (4.25) for all reviews
- **Advantages:** 
  - Extremely simple and fast (O(1) prediction)
  - Provides lower bound on performance
  - No overfitting risk
- **Disadvantages:**
  - Ignores all user, item, and content signals
  - Poor performance (Test RMSE: 1.329)
- **Complexity:** O(1) training and prediction

**2. User/Item Mean Models**
- **Approach:** Predict based on historical user or item average ratings
- **Advantages:**
  - Simple to implement and interpret
  - Captures user rating tendencies and item quality
  - Better than global mean (Test RMSE: ~1.37)
- **Disadvantages:**
  - Cold-start problem for new users/items (fallback to global mean)
  - No interaction between user and item signals
  - Cannot model temporal dynamics
- **Complexity:** O(n) training, O(1) prediction with hash tables

**3. Bias Model (α + bu + bi)**
- **Approach:** Decompose ratings into global mean + user bias + item bias
- **Mathematical Formulation:**
  - r̂ui = α + bu + bi
  - where α = global mean
  - bu = user bias (how much user u rates above/below average)
  - bi = item bias (how much item i is rated above/below average)
- **Advantages:**
  - Captures systematic user/item effects efficiently
  - Regularization (λ=10.0) prevents overfitting on rare users/items
  - Fast training with vectorized operations (~5 seconds)
  - Strong baseline (Test RMSE: 1.284)
- **Disadvantages:**
  - Linear model cannot capture user-item interactions
  - Still has cold-start issues
  - Ignores review text content
- **Complexity:** O(n × k) training for k iterations, O(1) prediction
- **Implementation:** We use alternating least squares with L2 regularization, converging in ~10 iterations

#### **Advanced Models:**

**4. TF-IDF + Ridge Regression (Content-Based)**
- **Approach:** Extract text features using TF-IDF, train Ridge regression
- **Feature Engineering:**
  - TF-IDF with 5,000 max features
  - Unigrams and bigrams (1-2 word phrases)
  - Min document frequency: 5 (removes ultra-rare words)
  - Max document frequency: 70% (removes overly common words)
  - Regularization: α=1.0
- **Advantages:**
  - **Best performing model** (Test RMSE: 0.935)
  - Captures sentiment and product-specific language
  - No cold-start problem for new users/items (uses text only)
  - Can identify positive/negative words
  - Generalizes well across rating values
- **Disadvantages:**
  - Computationally expensive (TF-IDF computation + large feature matrix)
  - Memory intensive (190k × 5k sparse matrix)
  - Doesn't leverage collaborative signals
  - Assumes text length and vocabulary are predictive
- **Complexity:** O(n × f) training where f=5,000 features, O(f) prediction
- **Key Insight:** High-rated reviews contain words like "great", "excellent", "perfect" while low-rated reviews contain "broken", "defective", "returned"

**5. Matrix Factorization (Latent Factor Model)**
- **Approach:** Learn latent representations of users and items via SGD
- **Mathematical Formulation:**
  - r̂ui = α + bu + bi + pu · qi
  - where pu = k-dimensional latent vector for user u
  - qi = k-dimensional latent vector for item i
  - k = 20 (number of latent factors)
- **Training:** Stochastic Gradient Descent over 20 epochs
  - Learning rate: η = 0.005
  - Regularization: λ = 0.02
  - Converges in ~60 seconds
- **Advantages:**
  - Captures latent user preferences and item characteristics
  - Models user-item interactions beyond simple biases
  - Better than bias model (Test RMSE: 1.283 vs 1.284)
  - Classic recommender system approach
- **Disadvantages:**
  - Severe cold-start problem (needs user/item history)
  - Hyperparameter sensitive (k, η, λ require tuning)
  - Risk of overfitting on popular items
  - Computationally expensive for large k
- **Complexity:** O(n × k × e) for e epochs, O(k) prediction
- **Implementation Challenge:** SGD requires careful tuning - too high learning rate causes divergence, too low requires many epochs

**6. Ensemble Model (Weighted Combination)**
- **Approach:** Combine predictions from Bias, TF-IDF, and Matrix Factorization
- **Weighting Strategy:** Weights proportional to 1/RMSE_validation
  - TF-IDF + Ridge: 0.457 (strongest weight)
  - Matrix Factorization: 0.273
  - Bias Model: 0.270
- **Advantages:**
  - Combines strengths of collaborative and content-based approaches
  - Robust to individual model failures
  - Second-best performance (Test RMSE: 1.057)
  - Reduces variance through averaging
- **Disadvantages:**
  - Increased complexity and computational cost
  - Requires training multiple models
  - Less interpretable than individual models
  - Doesn't improve beyond best component (TF-IDF)
- **Complexity:** Sum of component complexities

### 3.3 Code Walkthrough

  - See notebook for implementation. 

## Part 4 — Model Evaluation and Comparison

### 4.1 Context and Justification

We evaluate our models using multiple complementary metrics to understand different aspects of prediction quality:

**Primary Metrics:**
- **RMSE (Root Mean Squared Error):** Penalizes large errors more heavily, appropriate for rating prediction
- **MAE (Mean Absolute Error):** Interpretable in rating units (e.g., MAE=0.68 means average error of 0.68 stars)

**Secondary Metrics:**
- **Classification Accuracy:** After rounding to nearest integer (1-5), what percentage are exactly correct?
- **Confusion Matrix:** Shows systematic biases (e.g., do we over-predict 5-stars?)
- **R² (Coefficient of Determination):** Measures proportion of variance explained

**Why These Metrics?**

RMSE and MAE are appropriate because ratings are fundamentally continuous, we care about error magnitude, and they enable gradient-based optimization. Classification accuracy is less meaningful due to the 5-star positive skew.

**Test Set Results (Sorted by RMSE):**

| Model | RMSE | MAE | Accuracy | Improvement |
|-------|------|-----|----------|-------------|
| TF-IDF + Ridge | 0.935 | 0.676 | 52.9% | 29.7% |
| Ensemble | 1.057 | 0.801 | 39.6% | 20.5% |
| Matrix Fact. | 1.283 | 0.971 | 26.8% | 3.5% |
| Bias Model | 1.284 | 0.978 | 24.6% | 3.4% |
| Global Mean | 1.329 | 1.037 | 12.9% | 0.0% |

### 4.2 Discussing Baselines and Effectiveness of Our Models

**Key Finding 1: TF-IDF Dominates**

The text-based model achieves the best performance by a large margin (RMSE: 0.935), with 29.7% improvement over baseline. This demonstrates that review text contains rich sentiment signals that are highly predictive of ratings. The model successfully learns that words like "excellent" and "perfect" predict high ratings, while "broken" and "defective" predict low ratings.

**Key Finding 2: Cold-Start Resilience**

TF-IDF maintains consistent performance (~0.68 MAE) across all user types, from new users (0-1 reviews) to heavy users (20+ reviews). In contrast, collaborative methods degrade significantly for new users:

- New users: Bias Model MAE = 1.082 vs TF-IDF MAE = 0.681 (0.401 difference)
- Heavy users: Bias Model MAE = 0.834 vs TF-IDF MAE = 0.668 (0.166 difference)

This makes TF-IDF ideal for production systems where most users and items have limited history.

**Key Finding 3: Collaborative Methods Struggle**

Matrix Factorization and Bias Model barely beat the global mean baseline (RMSE ~1.28 vs 1.33). This is surprising but explained by:
- Extreme sparsity: 190k ratings across 87k users × 45k items (>99.9% sparse)
- Temporal shift: Test set contains newer products not seen in training
- Cold-start dominance: 60% of test users and 45% of test items are new/rare

**Key Finding 4: Ensemble Doesn't Help**

The ensemble achieves second-best performance (RMSE: 1.057) but doesn't beat TF-IDF alone. This occurs because TF-IDF dominates the weighting (45.7%) and adding weaker models slightly hurts performance. Ensembles work best when component models have similar accuracy but different error patterns - here, TF-IDF is simply superior.

**Stratified Performance:**

Performance varies systematically across data segments:
- 5-star reviews are easiest to predict (most common, clear sentiment)
- 1-2 star reviews are harder (less common, more variable language)
- 3-star reviews are most challenging (ambiguous sentiment)

**Residual Analysis:**

The TF-IDF model shows excellent calibration:
- Mean residual: -0.003 (near zero, unbiased)
- Residual std: 0.934 (matches RMSE)
- Errors follow approximately normal distribution
- Slight regression to mean (underpredicts low ratings, overpredicts high ratings)

### 4.3 Code Walkthrough

Our evaluation framework creates comprehensive comparisons across all models and metrics. We visualize performance using multiple complementary views:

1. **Grouped bar charts:** Compare RMSE/MAE/Accuracy across train/val/test
2. **Mean predictions with error bars:** Show calibration and uncertainty
3. **Residual diagnostics:** Confirm model assumptions and identify biases
4. **Stratified analysis:** Reveal cold-start and popularity effects

The code uses pandas DataFrames for structured comparison and matplotlib/seaborn for visualizations. See notebook for detailed implementation.