# Movie Recommendation System

## Overview
This repository contains a **Movie Recommendation System** built using various collaborative filtering and content-based filtering techniques. The project demonstrates the usage of the **MovieLens 100k dataset** for predicting user preferences and analyzing movie trends.

The system utilizes libraries such as:
- **pandas** and **numpy** for data manipulation.
- **matplotlib** and **seaborn** for data visualization.
- **scikit-surprise** for collaborative filtering.
- **nltk** for natural language processing.
- **scikit-learn** for content-based filtering.

---

## Features
### Data Analysis
1. **Rating Distribution**:
   - Visualizes the distribution of user ratings.
2. **User Activity Distribution**:
   - Highlights how active users are in providing ratings.
3. **Movie Popularity**:
   - Identifies popular movies based on the number of ratings.
4. **Genre Analysis**:
   - Displays the distribution of movies across different genres.

### Collaborative Filtering
Implemented collaborative filtering algorithms using the `surprise` library:
1. **SVD (Singular Value Decomposition)**
2. **KNN With Means**
3. **BaselineOnly**
4. **KNN Basic (Cosine Similarity)**
5. **NMF (Non-Negative Matrix Factorization)**

### Content-Based Filtering
- **TF-IDF Vectorization**:
  - Analyzes the similarity between movie titles using **cosine similarity**.

### Movie Recommendations
- A function `get_movie_recommendations(title)` suggests 10 movies similar to the input title based on cosine similarity.

---

## Dataset
The project uses the **MovieLens 100k Dataset**, which contains:
- **u.data**: User-item ratings.
- **u.item**: Movie metadata including title, release date, genres, etc.

### Data Sources:
- [Ratings Data](http://files.grouplens.org/datasets/movielens/ml-100k/u.data)
- [Movies Data](http://files.grouplens.org/datasets/movielens/ml-100k/u.item)

---

## Installation
To run this project, install the following Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-surprise nltk
```

Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

---

## Usage
### 1. Data Visualization
Run the script to visualize:
- Rating distributions.
- User activity levels.
- Movie popularity.
- Genre distributions.

### 2. Model Training
Collaborative filtering models trained and evaluated include:
- **SVD**: RMSE and MAE are computed.
- **KNN With Means**: Pearson similarity used for predictions.
- **BaselineOnly**: Predicts ratings based on global averages.
- **KNN Basic**: Uses cosine similarity.
- **NMF**: Factorizes the user-item matrix.

### 3. Recommendation
Generate movie recommendations:
```python
print(get_movie_recommendations("Star Wars (1977)"))
```

### 4. Evaluation
The models are evaluated based on:
- **RMSE (Root Mean Square Error)**
- **MAE (Mean Absolute Error)**

Comparative bar plots of RMSE and MAE scores are generated for all models.

---
