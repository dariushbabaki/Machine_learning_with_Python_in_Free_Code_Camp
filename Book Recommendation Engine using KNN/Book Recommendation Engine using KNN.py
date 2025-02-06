import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
books = pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1')
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1')

# Data preprocessing
user_ratings = ratings.groupby('User-ID').size()  # Count ratings per user
books_ratings = ratings.groupby('ISBN').size()  # Count ratings per book

# Filter users and books with enough ratings
users_with_enough_ratings = user_ratings[user_ratings >= 200].index
books_with_enough_ratings = books_ratings[books_ratings >= 100].index

# Filter ratings to include only users and books with enough ratings
ratings_filtered = ratings[ratings['User-ID'].isin(users_with_enough_ratings)]
ratings_filtered = ratings_filtered[ratings_filtered['ISBN'].isin(books_with_enough_ratings)]

# Create a feature matrix for books (each row is a book, columns are users and ratings)
book_features = ratings_filtered.pivot_table(index='ISBN', columns='User-ID', values='Book-Rating').fillna(0)
book_matrix = book_features.values  # Convert to numpy array

# Train KNN model
knn = NearestNeighbors(n_neighbors=6, metric='cosine')  # Using cosine similarity
knn.fit(book_matrix)

# Define function to get book recommendations
def get_recommends(book_title):
    # Get the ISBN of the input book
    book_isbn = books[books['Book-Title'] == book_title]['ISBN'].values[0]
    
    # Find the index of the book in the feature matrix
    book_index = book_features.index.get_loc(book_isbn)
    
    # Find the nearest neighbors (similar books)
    distances, indices = knn.kneighbors(book_matrix[book_index].reshape(1, -1), n_neighbors=6)
    
    # Create a list of recommended books and their distances
    recommended_books = []
    for i, idx in enumerate(indices[0][1:]):  # Skip the first book as it's the same one
        recommended_book_title = books[books['ISBN'] == book_features.index[idx]]['Book-Title'].values[0]
        recommended_books.append([recommended_book_title, distances[0][i + 1]])

    return [book_title, recommended_books]

# Test the function
print(get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))"))
