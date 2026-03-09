import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Multiply, Concatenate, Dense
from tensorflow.keras.optimizers import Adam
import os

# Set page config
st.set_page_config(page_title="Anime Recommender", page_icon="🎬", layout="wide")

# App title and description
st.title("🎬 Anime Recommendation System")
st.markdown("""
Welcome to the AI-powered Anime Recommender! Enter your User ID, select a recommendation method, and discover your next favorite anime!
""")

# Load data with caching
@st.cache_data
def load_data():
    ratings = pd.read_csv('1lakhratings.csv')
    anime_df = pd.read_csv('anime.csv')
    return ratings, anime_df

ratings_orig, anime_df = load_data()

# Model-specific functions
@st.cache_resource
def get_svd_recommendations(user_id, _ratings_df, _anime_df):
    # Process ratings
    ratings = _ratings_df.copy()
    ratings['rating'] = ratings['rating'].replace(-1, np.nan)
    
    # Pivot matrix
    user_anime_matrix = ratings.pivot(index='user_id', columns='anime_id', values='rating')
    user_ratings_mean = user_anime_matrix.mean(axis=1)
    user_anime_matrix_normalized = user_anime_matrix.sub(user_ratings_mean, axis=0)
    user_anime_matrix_filled = user_anime_matrix_normalized.fillna(0)
    
    # SVD
    user_anime_sparse = csr_matrix(user_anime_matrix_filled.values)
    num_factors = 50
    U, sigma, Vt = svds(user_anime_sparse, k=num_factors)
    sigma = np.diag(sigma)
    
    predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.values.reshape(-1, 1)
    predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_anime_matrix.columns, index=user_anime_matrix.index)
    
    # Check if user exists
    fallback_used = False
    original_id = user_id
    if user_id not in predicted_ratings_df.index:
        user_id = np.random.choice(predicted_ratings_df.index)
        fallback_used = True
        
    user_pred_ratings = predicted_ratings_df.loc[user_id]
    rated_animes = user_anime_matrix.loc[user_id].dropna().index
    unrated_animes = user_pred_ratings.drop(rated_animes, errors='ignore')
    
    top_animes = unrated_animes.sort_values(ascending=False).head(10)
    recommended_anime_ids = top_animes.index.tolist()
    
    reccs = _anime_df[_anime_df['anime_id'].isin(recommended_anime_ids)][['anime_id', 'name', 'genre', 'type', 'rating']]
    return reccs, user_id, fallback_used

@st.cache_resource
def get_cf_sgd_recommendations(user_id, _ratings_df, _anime_df):
    ratings = _ratings_df[_ratings_df['rating'] != -1].copy()
    unique_users = ratings['user_id'].unique()
    unique_animes = ratings['anime_id'].unique()
    num_users, num_animes = len(unique_users), len(unique_animes)
    
    user_id_to_index = {uid: idx for idx, uid in enumerate(unique_users)}
    anime_id_to_index = {aid: idx for idx, aid in enumerate(unique_animes)}
    
    rated_list = [(user_id_to_index[row['user_id']], anime_id_to_index[row['anime_id']], row['rating']) 
                  for _, row in ratings.iterrows()]
    
    K, eta, lambda_reg, epochs = 20, 0.01, 0.1, 10 # Reduced epochs for faster app serving
    P = np.random.normal(0, 0.1, (num_users, K))
    Q = np.random.normal(0, 0.1, (num_animes, K))
    
    for epoch in range(epochs):
        np.random.shuffle(rated_list)
        for u, i, r in rated_list:
            pred = np.dot(P[u], Q[i])
            error = r - pred
            P[u] += eta * (error * Q[i] - lambda_reg * P[u])
            Q[i] += eta * (error * P[u] - lambda_reg * Q[i])
            
    predicted_ratings_cf = np.dot(P, Q.T)
    
    fallback_used = False
    if user_id not in user_id_to_index:
        user_id = np.random.choice(unique_users)
        fallback_used = True
        
    u_idx = user_id_to_index[user_id]
    rated_animes = ratings[ratings['user_id'] == user_id]['anime_id'].tolist()
    rated_indices = [anime_id_to_index[aid] for aid in rated_animes if aid in anime_id_to_index]
    
    user_pred = predicted_ratings_cf[u_idx].copy()
    user_pred[rated_indices] = -np.inf
    
    top_indices = np.argsort(user_pred)[-10:][::-1]
    recommended_anime_ids = [unique_animes[idx] for idx in top_indices]
    
    reccs = _anime_df[_anime_df['anime_id'].isin(recommended_anime_ids)][['anime_id', 'name', 'genre', 'type', 'rating']]
    return reccs, user_id, fallback_used

@st.cache_resource
def get_ncf_recommendations(user_id, _ratings_df, _anime_df):
    ratings = _ratings_df[_ratings_df['rating'] != -1].copy()
    unique_users = ratings['user_id'].unique()
    unique_animes = ratings['anime_id'].unique()
    num_users, num_animes = len(unique_users), len(unique_animes)
    
    user_id_to_index = {uid: idx for idx, uid in enumerate(unique_users)}
    anime_id_to_index = {aid: idx for idx, aid in enumerate(unique_animes)}
    
    users = ratings['user_id'].map(user_id_to_index).values
    animes = ratings['anime_id'].map(anime_id_to_index).values
    ratings_values = ratings['rating'].values
    
    # Build Model
    embedding_size = 32
    user_input = Input(shape=(1,), name='user_input')
    anime_input = Input(shape=(1,), name='anime_input')
    
    gmf_user_evt = Embedding(num_users, embedding_size)(user_input)
    gmf_user_evt = Flatten()(gmf_user_evt)
    gmf_anime_evt = Embedding(num_animes, embedding_size)(anime_input)
    gmf_anime_evt = Flatten()(gmf_anime_evt)
    gmf_product = Multiply()([gmf_user_evt, gmf_anime_evt])
    
    mlp_user_evt = Embedding(num_users, embedding_size)(user_input)
    mlp_user_evt = Flatten()(mlp_user_evt)
    mlp_anime_evt = Embedding(num_animes, embedding_size)(anime_input)
    mlp_anime_evt = Flatten()(mlp_anime_evt)
    mlp_concat = Concatenate()([mlp_user_evt, mlp_anime_evt])
    mlp_dense1 = Dense(64, activation='relu')(mlp_concat)
    mlp_dense2 = Dense(32, activation='relu')(mlp_dense1)
    
    neu_concat = Concatenate()([gmf_product, mlp_dense2])
    prediction = Dense(1, activation='linear')(neu_concat)
    
    model = Model(inputs=[user_input, anime_input], outputs=prediction)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    model.fit([users, animes], ratings_values, epochs=1, batch_size=256, verbose=0)
    
    fallback_used = False
    if user_id not in user_id_to_index:
        user_id = np.random.choice(unique_users)
        fallback_used = True
    
    user_idx = user_id_to_index[user_id]
    user_rated_animes = ratings[ratings['user_id'] == user_id]['anime_id'].tolist()
    user_rated_indices = {anime_id_to_index[aid] for aid in user_rated_animes if aid in anime_id_to_index}
    
    unrated_indices = [idx for idx in range(num_animes) if idx not in user_rated_indices]
    user_array = np.array([user_idx] * len(unrated_indices))
    anime_array = np.array(unrated_indices)
    
    predicted_ratings_ncf = model.predict([user_array, anime_array], batch_size=512, verbose=0).flatten()
    top_indices = np.argsort(predicted_ratings_ncf)[-10:][::-1]
    recommended_indices = [unrated_indices[i] for i in top_indices]
    recommended_anime_ids = [unique_animes[idx] for idx in recommended_indices]
    
    reccs = _anime_df[_anime_df['anime_id'].isin(recommended_anime_ids)][['anime_id', 'name', 'genre', 'type', 'rating']]
    return reccs, user_id, fallback_used

# Sidebar UI
st.sidebar.header("User Selection")
user_input = st.sidebar.number_input("Enter User ID", min_value=1, value=52969, step=1)
algorithm = st.sidebar.selectbox(
    "Select Recommendation Algorithm",
    ("Matrix Factorization (SVD)", "Collaborative Filtering (SGD)", "Neural Collaborative Filtering (NCF)")
)

if st.sidebar.button("Get Recommendations"):
    with st.spinner(f"Running {algorithm}..."):
        if algorithm == "Matrix Factorization (SVD)":
            reccs, used_id, fallback = get_svd_recommendations(user_input, ratings_orig, anime_df)
        elif algorithm == "Collaborative Filtering (SGD)":
            reccs, used_id, fallback = get_cf_sgd_recommendations(user_input, ratings_orig, anime_df)
        else: # Neural Collaborative Filtering (NCF)
            reccs, used_id, fallback = get_ncf_recommendations(user_input, ratings_orig, anime_df)
            
        if fallback:
            st.warning(f"User ID {user_input} not found in dataset. Showing recommendations for a random user: {used_id}")
        else:
            st.success(f"Top 10 Recommendations for User {used_id}")
            
        st.dataframe(reccs, use_container_width=True)
else:
    st.info("👈 Use the sidebar to enter a User ID and choose an algorithm to get started!")
    
    # Sample display
    st.header("Top Rated Anime (Overall)")
    top_rated = anime_df.sort_values(by='rating', ascending=False).head(5)
    st.dataframe(top_rated[['name', 'genre', 'rating']], use_container_width=True)
