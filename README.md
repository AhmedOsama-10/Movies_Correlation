# Movies Correlation Project

## Overview

This project explores the relationship between various movie features, such as budget, gross revenue, and votes. The dataset undergoes cleaning, imputation, and visualization to identify key patterns.

## Dataset

The dataset contains movie-related features such as:

- **Budget**: The cost to produce the movie
- **Gross Revenue**: The total earnings
- **Votes**: The number of audience votes
- **Year**: Release year
- **Score**: IMDb rating
- **Other metadata**: Director, writer, star, country, etc.

## Installation

To run this project, install the required dependencies:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Data Preprocessing

### 1. Importing Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
```

### 2. Loading the Dataset

```python
data = pd.read_csv('movies.csv')
data.head()
```

### 3. Handling Missing Values

We use the **KNN Imputer** to fill missing values for numerical columns (`budget`, `votes`, `gross`).

```python
selected_columns = ['budget', 'votes', 'gross']
data_subset = data[selected_columns]
imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(data_subset)
data_imputed = pd.DataFrame(data_imputed, columns=selected_columns)
data['budget'] = data_imputed['budget']
```

### 4. Dropping Irrelevant Features

We drop the `released` column due to excessive missing values.

```python
data.drop(columns=['released'], inplace=True)
```

### 5. Data Transformation

Convert numerical columns to integer format for better visualization.

```python
data[['votes', 'budget', 'gross', 'runtime']] = data[['votes', 'budget', 'gross', 'runtime']].astype('int')
```

## Exploratory Data Analysis

### 1. Correlation Matrix

To identify relationships between features:

```python
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()
```

### 2. Budget vs. Gross Revenue

A scatter plot with a regression line:

```python
sns.lmplot(x='budget', y='gross', data=data, aspect=2, height=6, line_kws={'color': 'blue'})
plt.xlabel('Budget')
plt.ylabel('Gross Revenue')
plt.title('Budget vs Gross with Regression Line')
plt.show()
```

## Results

- **High correlation** between `budget` and `gross revenue`.
- **Votes and gross revenue** also show a strong correlation.
- The correlation matrix helps identify the strongest predictors for movie success.

## Conclusion

This project provides insights into the financial aspects of movie production. Future work may include:

- More advanced feature engineering.
- Exploring additional datasets.
- Implementing predictive models for movie success.

## License

This project is licensed under the MIT License.

