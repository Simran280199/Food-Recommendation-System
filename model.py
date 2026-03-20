import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
food = pd.read_csv("1662574418893344.csv")
ratings = pd.read_csv("ratings.csv")

# Cleaning
food.dropna(inplace=True)
ratings.dropna(inplace=True)

food['Describe'] = food['Describe'].astype(str)
food['Name'] = food['Name'].astype(str).str.lower()

# Content-based
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(food['Describe'])

content_similarity = cosine_similarity(tfidf_matrix)

# Collaborative
pivot = ratings.pivot_table(index='User_ID', columns='Food_ID', values='Rating')
pivot.fillna(0, inplace=True)

collab_similarity = cosine_similarity(pivot.T)

# Hybrid function
def hybrid_recommend(food_name):
    food_name = food_name.lower()

    matches = food[food['Name'].str.contains(food_name)]

    if matches.empty:
        return ["Food not found"]

    idx = matches.index[0]

    content_scores = list(enumerate(content_similarity[idx]))
    collab_scores = list(enumerate(collab_similarity[idx]))

    hybrid_scores = {}

    for i, score in content_scores:
        hybrid_scores[i] = score * 0.6

    for i, score in collab_scores:
        if i in hybrid_scores:
            hybrid_scores[i] += score * 0.4

    sorted_scores = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in sorted_scores[1:6]:
        recommendations.append(food.iloc[i[0]]['Name'])

    return recommendations