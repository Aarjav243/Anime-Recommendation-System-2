#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Matrix factorization
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split

#loading the dataset and naming it as ratings
ratings = pd.read_csv('1lakhratings.csv')

#removing unrated animes to train the data 
ratings['rating'] = ratings['rating'].replace(-1, np.nan)

#creating a matrix with user id's as rows, anime id's as columns and each cell representing a rating
user_anime_matrix = ratings.pivot(index='user_id', columns='anime_id', values='rating')

#normalizing the rating (first calculating the average ratings across each row and then subtracting it from each rating)
#purpose :- It will brings every rating close to 0 which normalizes the ratings which makes it easier for model to learn
user_ratings_mean = user_anime_matrix.mean(axis=1)
user_anime_matrix_normalized = user_anime_matrix.sub(user_ratings_mean, axis=0)

#filling the nan values with 0 making it easier for model to learn
user_anime_matrix_filled = user_anime_matrix_normalized.fillna(0)

#making a sparse matrix with the values of our user_anime_matrix_table
user_anime_sparse = csr_matrix(user_anime_matrix_filled.values)

#applying SVD (It is method used in algrebra to split a matrix into 3 matrices)
#num factors are underlying characterstics that captures the essential traits of users and animes with these factors 
#U : A matrix with users as rows and columns made up of num-factors 
#sigma :- A 1D matrix with num factors 
#V : A matrix with anime as rows and columns made up of num-factors
#Vt : Transpose of V
num_factors = 50
U, sigma, Vt = svds(user_anime_sparse, k=num_factors)
sigma = np.diag(sigma)

# Reconstruct the predicted ratings matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.values.reshape(-1, 1)
#predicting the ratings first by doing a dot product first U and sigmna and their dot prodcut with vt and adding the mean(-1,1 reshapes the matrix to shape of U, Sigma,VT)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_anime_matrix.columns, index=user_anime_matrix.index)#creating a dataframe of predicted ratings just like user_anime_matrix

# Function to get top-N recommendations for a user
def get_recommendations(user_id, top_n=10):
    if user_id not in predicted_ratings_df.index:
        return "User ID not found."

    #get user's predicted ratings with loc functiom from predicted ratings table
    user_pred_ratings = predicted_ratings_df.loc[user_id]

   #getting animes already rated by user from user_anime_matrix
    rated_animes = user_anime_matrix.loc[user_id].dropna().index

    #filtering out rated animes 
    unrated_animes = user_pred_ratings.drop(rated_animes)

    #get top N anime id's in descending order
    top_animes = unrated_animes.sort_values(ascending=False).head(top_n)
    recommended_anime_ids = top_animes.index.tolist()

    #mapping with anime dataset to get the names of anime along with their id's
    anime_df = pd.read_csv('anime.csv')
    recommended_names = anime_df[anime_df['anime_id'].isin(recommended_anime_ids)][['anime_id', 'name']]
    return recommended_names

#testing with a sample user id 
user_id = 52969
if user_id not in predicted_ratings_df.index:
    print(f"User ID {user_id} not found. Choosing a valid random user ID instead.")
    user_id = np.random.choice(predicted_ratings_df.index)

recommendations = get_recommendations(user_id)
print(f"\nTop 10 recommendations for user {user_id}:")
print(recommendations)


# In[ ]:


#Collaborative filtering
import pandas as pd
import numpy as np

#setting random seed for reproducibility
np.random.seed(42)

# Loading the rating dataset
ratings = pd.read_csv('1lakhratings.csv')

# Filtering out unrated animes (rating = -1)
ratings = ratings[ratings['rating'] != -1]

#getting unique users and animes 
unique_users = ratings['user_id'].unique()
unique_animes = ratings['anime_id'].unique()
#getting the lenght of those unique users and animes 
num_users = len(unique_users)
num_animes = len(unique_animes)

#creating dictionaries to map user id to index 
user_id_to_index = {uid: idx for idx, uid in enumerate(unique_users)}#enumerate function here creates tuples of (index, user_id) and then we are converting them into dictionary
anime_id_to_index = {aid: idx for idx, aid in enumerate(unique_animes)}

# Create list of (u_idx, a_idx, rating) for rated animes
rated_list = [
    (user_id_to_index[row['user_id']], anime_id_to_index[row['anime_id']], row['rating'])
    for _, row in ratings.iterrows()
]

#rated list is a list which has tuples of user_id, anime_id and rating 
#our hyperparameters
K = 20  #number of latent factors
eta = 0.01#learning rate of model
lambda_reg = 0.1#It is a regularization parameter that prevents overfitting(model sometimes learn about training data) by penalizing the model
epochs = 50  # Number of training rounds 

#Intializing latent factor matrices 
#here 0 is our mean and 0.1 is our std and (num_users, K) is our size of matrix , basically creating a standardized matrix of num_users * latent factors
P = np.random.normal(0, 0.1, (num_users, K))


Q = np.random.normal(0, 0.1, (num_animes, K))#standardized matrix of anime_id * latent factors 

#training :- It updates P(user_matrix) and Q(anime_matrix) based on user and anime ratings respectively 
for epoch in range(epochs):
    np.random.shuffle(rated_list)#shuffles the entries which is good for training 
    for u, i, r in rated_list:
        # u = Index of the user 
        #i = Index of anime 
        #r = actual rating 
        pred = np.dot(P[u], Q[i])
        error = r - pred
        P[u] += eta * (error * Q[i] - lambda_reg * P[u])
        Q[i] += eta * (error * P[u] - lambda_reg * Q[i])
        #error * Q[i] : this parts calculates user features based on error and based on that specific anime
        #lambda_reg * P[u] : This parts prevents overfitting 

        #compute predicted ratings matrix 

# Compute predicted ratings matrix
predicted_ratings_cf = np.dot(P, Q.T)

# Function to get top-N recommendations for a user
def get_recommendations(user_id, top_n=10):
    if user_id not in user_id_to_index:
        return "User ID not found."

    u_idx = user_id_to_index[user_id]#gets the index of that particular user_id 

     #getting animes rated by that particular user
    rated_animes = ratings[ratings['user_id'] == user_id]['anime_id'].tolist()
    rated_indices = [anime_id_to_index[aid] for aid in rated_animes if aid in anime_id_to_index]

     #getting predicted ratings for all animes of that particular user 
    user_pred = predicted_ratings_cf[u_idx].copy()

    # Set rated animes to -inf to exclude them from recommendations
    user_pred[rated_indices] = -np.inf

     #getting indices of those top-10 animes in desceding order
    top_indices = np.argsort(user_pred)[-top_n:][::-1]
    recommended_anime_ids = [unique_animes[idx] for idx in top_indices]

     #now getting the names of those anime_id's from anime csv
    anime_df = pd.read_csv('anime.csv')
    recommended_names = anime_df[anime_df['anime_id'].isin(recommended_anime_ids)][['anime_id', 'name']]
    return recommended_names

#testing with a sample user id 
user_id = 52969
if user_id not in user_id_to_index:
    print(f"User ID {user_id} not found. Choosing a valid random user ID instead.")
    user_id = np.random.choice(unique_users)

recommendations = get_recommendations(user_id)
print(f"Top 10 recommendations for user {user_id}:")
print(recommendations)


# In[ ]:


#neural collaborative filtering 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Multiply, Concatenate, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Load and preprocess rating dataset
ratings = pd.read_csv('1lakhratings.csv')
ratings = ratings[ratings['rating'] != -1]  # Filter out unrated entries

# Get unique users and animes
unique_users = ratings['user_id'].unique()
unique_animes = ratings['anime_id'].unique()
num_users = len(unique_users)
num_animes = len(unique_animes)

# Create ID-to-index mappings
user_id_to_index = {uid: idx for idx, uid in enumerate(unique_users)}
anime_id_to_index = {aid: idx for idx, aid in enumerate(unique_animes)}

# Prepare training data
users = ratings['user_id'].map(user_id_to_index).values#it replaces the index with user_id 
animes = ratings['anime_id'].map(anime_id_to_index).values#it replaces the index with anime_id
ratings_values = ratings['rating'].values#.values converts the values into numpy array 

# Split into training and validation sets with a fixed random state
train_users, val_users, train_animes, val_animes, train_ratings, val_ratings = train_test_split(
    users, animes, ratings_values, test_size=0.2, random_state=42
)

# Define NCF model
embedding_size = 32#embedding_size is like latent factors , it gives us 32 factors in the form of numbers which looks for patterns within the data 

# Input layers
user_input = Input(shape=(1,), name='user_input')#defining the layers of one dimension
anime_input = Input(shape=(1,), name='anime_input')

# GMF pathway :- Making understand linear relationships through this 
gmf_user_embedding = Embedding(num_users, embedding_size, name='gmf_user_embedding')(user_input)#a 3d array of [user_id, 1 , embedding_size]
gmf_user_embedding = Flatten()(gmf_user_embedding)#flatten is used to convert this 3d array into a 2d array of [user_id, embedding_size]
gmf_anime_embedding = Embedding(num_animes, embedding_size, name='gmf_anime_embedding')(anime_input)#same for anime_id array as well 
gmf_anime_embedding = Flatten()(gmf_anime_embedding)
gmf_product = Multiply(name='gmf_product')([gmf_user_embedding, gmf_anime_embedding])#we want model to understand linear data, therefore we use multiply function here 

# MLP pathway :- Model can learn about non linear data here 
mlp_user_embedding = Embedding(num_users, embedding_size, name='mlp_user_embedding')(user_input)
mlp_user_embedding = Flatten()(mlp_user_embedding)
mlp_anime_embedding = Embedding(num_animes, embedding_size, name='mlp_anime_embedding')(anime_input)
mlp_anime_embedding = Flatten()(mlp_anime_embedding)
mlp_concat = Concatenate(name='mlp_concat')([mlp_user_embedding, mlp_anime_embedding])#using concacenating so that model can understand about non linear and complex data
mlp_dense1 = Dense(64, activation='relu', name='mlp_dense1')(mlp_concat)#there are 64 neurons here, each has a different learning pattern 
mlp_dense2 = Dense(32, activation='relu', name='mlp_dense2')(mlp_dense1)#we are reducing the no of neurons here, so that model can learn can focus on most important features 

# Combine GMF and MLP
neu_concat = Concatenate(name='neu_concat')([gmf_product, mlp_dense2])#by  combining both the features model can learn about linear and complex data 
prediction = Dense(1, activation='linear', name='prediction')(neu_concat)

# Build and compile model
model = Model(inputs=[user_input, anime_input], outputs=prediction)
model.compile(optimizer=Adam(), loss='mse')

# Train the model with a fixed random state
model.fit(
    [train_users, train_animes], train_ratings,
    validation_data=([val_users, val_animes], val_ratings),
    epochs=1,
    batch_size=256,
    verbose=1
)

# Recommendation function
def get_recommendations(user_id, top_n=10):
    if user_id not in user_id_to_index:
        return "User ID not found."
    
    user_idx = user_id_to_index[user_id]#getting the user id of that particular user 
    user_rated_animes = ratings[ratings['user_id'] == user_id]['anime_id'].tolist()#getting all the animes rated by that user and converting it into a list 
    user_rated_indices = {anime_id_to_index[aid] for aid in user_rated_animes if aid in anime_id_to_index}#getting indices of all those anime id's which are rated by the user 
    
    #getting all the unrated animes 
    unrated_indices = [idx for idx in range(num_animes) if idx not in user_rated_indices]
    
    #user array = [5, 5, 5, 5, 5] if user_idx = 5 and len(unrated_indices) = 5, it works in this way 
    user_array = np.array([user_idx] * len(unrated_indices))
    anime_array = np.array(unrated_indices)#converting into a numpy for easier computations 
    predicted_ratings = model.predict([user_array, anime_array], batch_size=256).flatten()#now predicting the ratings of all those unrated entries using the model we have trained 
    
    # Get top N recommendations
    top_indices = np.argsort(predicted_ratings)[-top_n:][::-1]#arranging top 10 anime_id's in descending order 
    recommended_indices = [unrated_indices[i] for i in top_indices]
    recommended_anime_ids = [unique_animes[idx] for idx in recommended_indices]
    
    # Map to anime names
    anime_df = pd.read_csv('anime.csv')
    recommended_names = anime_df[anime_df['anime_id'].isin(recommended_anime_ids)][['anime_id', 'name']]
    return recommended_names

# Test with a sample user
user_id = 52969
if user_id not in user_id_to_index:
    print(f"User ID {user_id} not found. Choosing a valid random user ID instead.")
    user_id = np.random.choice(unique_users)

recommendations = get_recommendations(user_id)
print(f"\nTop 10 recommendations for user {user_id}:")
print(recommendations)


# In[ ]:


#computing metrics like rmse and mae to compare the 4 methods 
#we had to divide the datasets into training and testing datasets for compuation of rmse and mae
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Multiply, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Load data (common across all methods)
ratings = pd.read_csv('1lakhratings.csv')
ratings = ratings[ratings['rating'] != -1]  # Filter unrated entries

# Split into train and test sets
train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)


# In[ ]:


# --- Matrix Factorization using SVD ---
#most of the code for each of the recommendation technique is same as the codes we used for recommendations in next 4 codes 
user_anime_matrix = train_df.pivot(index='user_id', columns='anime_id', values='rating')
user_ratings_mean = user_anime_matrix.mean(axis=1)
user_anime_matrix_normalized = user_anime_matrix.sub(user_ratings_mean, axis=0)
user_anime_matrix_filled = user_anime_matrix_normalized.fillna(0)
user_anime_sparse = csr_matrix(user_anime_matrix_filled.values)
num_factors = 50
U, sigma, Vt = svds(user_anime_sparse, k=num_factors)
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.values.reshape(-1, 1)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_anime_matrix.columns, index=user_anime_matrix.index)

def get_recommendations_svd(user_id, train_df, top_n=10):
    if user_id not in predicted_ratings_df.index:
        return []
    user_pred_ratings = predicted_ratings_df.loc[user_id]
    rated_animes = train_df[train_df['user_id'] == user_id]['anime_id'].tolist()
    unrated_animes = user_pred_ratings.drop(rated_animes, errors='ignore')
    top_animes = unrated_animes.sort_values(ascending=False).head(top_n)
    return top_animes.index.tolist()

def predict_svd(user_id, anime_id):
    if user_id in predicted_ratings_df.index and anime_id in predicted_ratings_df.columns:
        return predicted_ratings_df.loc[user_id, anime_id]
    return 0 


# In[ ]:


# --- Collaborative Filtering with SGD ---
unique_users = ratings['user_id'].unique()
unique_animes = ratings['anime_id'].unique()
user_id_to_index = {uid: idx for idx, uid in enumerate(unique_users)}
anime_id_to_index = {aid: idx for idx, aid in enumerate(unique_animes)}
num_users, num_animes = len(unique_users), len(unique_animes)

rated_list = [
    (user_id_to_index[row['user_id']], anime_id_to_index[row['anime_id']], row['rating'])
    for _, row in train_df.iterrows()
]
K, eta, lambda_reg, epochs = 20, 0.001, 0.1, 50
P = np.random.normal(0, 0.1, (num_users, K))
Q = np.random.normal(0, 0.1, (num_animes, K))
for epoch in range(epochs):
    np.random.shuffle(rated_list)
    for u, i, r in rated_list:
        pred = np.dot(P[u], Q[i])
        error = r - pred
        P[u] += eta * (error * Q[i] - lambda_reg * P[u])
        Q[i] += eta * (error * P[u] - lambda_reg * Q[i])
predicted_ratings_sgd = np.dot(P, Q.T)

def predict_sgd(user_id, anime_id):
    u_idx = user_id_to_index.get(user_id, -1)
    a_idx = anime_id_to_index.get(anime_id, -1)
    if u_idx == -1 or a_idx == -1:
        return 0
    return predicted_ratings_sgd[u_idx, a_idx]

def get_recommendations_sgd(user_id, train_df, top_n=10):
    if user_id not in user_id_to_index:
        return []
    u_idx = user_id_to_index[user_id]
    rated_animes = train_df[train_df['user_id'] == user_id]['anime_id'].tolist()
    rated_indices = [anime_id_to_index[aid] for aid in rated_animes if aid in anime_id_to_index]
    user_pred = predicted_ratings_sgd[u_idx].copy()
    user_pred[rated_indices] = -np.inf
    top_indices = np.argsort(user_pred)[-top_n:][::-1]
    return [unique_animes[idx] for idx in top_indices]


# In[ ]:


# --- Neural Collaborative Filtering (NCF) ---
users = train_df['user_id'].map(user_id_to_index).values
animes = train_df['anime_id'].map(anime_id_to_index).values
ratings_values = train_df['rating'].values

embedding_size = 32
user_input = Input(shape=(1,))
anime_input = Input(shape=(1,))
gmf_user_embedding = Embedding(num_users, embedding_size)(user_input)
gmf_user_embedding = Flatten()(gmf_user_embedding)
gmf_anime_embedding = Embedding(num_animes, embedding_size)(anime_input)
gmf_anime_embedding = Flatten()(gmf_anime_embedding)
gmf_product = Multiply()([gmf_user_embedding, gmf_anime_embedding])
mlp_user_embedding = Embedding(num_users, embedding_size)(user_input)
mlp_user_embedding = Flatten()(mlp_user_embedding)
mlp_anime_embedding = Embedding(num_animes, embedding_size)(anime_input)
mlp_anime_embedding = Flatten()(mlp_anime_embedding)
mlp_concat = Concatenate()([mlp_user_embedding, mlp_anime_embedding])
mlp_dense1 = Dense(64, activation='relu')(mlp_concat)
mlp_dense2 = Dense(32, activation='relu')(mlp_dense1)
neu_concat = Concatenate()([gmf_product, mlp_dense2])
prediction = Dense(1, activation='linear')(neu_concat)
model = Model(inputs=[user_input, anime_input], outputs=prediction)
model.compile(optimizer=Adam(), loss='mse')
model.fit([users, animes], ratings_values, epochs=10, batch_size=256, verbose=1)

def predict_ncf(user_id, anime_id):
    u_idx = user_id_to_index.get(user_id, -1)
    a_idx = anime_id_to_index.get(anime_id, -1)
    if u_idx == -1 or a_idx == -1:
        return 0
    return model.predict([np.array([u_idx]), np.array([a_idx])], verbose=0)[0][0]

def get_recommendations_ncf(user_id, train_df, top_n=10):
    u_idx = user_id_to_index.get(user_id, -1)
    if u_idx == -1:
        return []
    rated_animes = train_df[train_df['user_id'] == user_id]['anime_id'].tolist()
    rated_indices = {anime_id_to_index[aid] for aid in rated_animes if aid in anime_id_to_index}
    unrated_indices = [idx for idx in range(num_animes) if idx not in rated_indices]
    user_array = np.array([u_idx] * len(unrated_indices))
    anime_array = np.array(unrated_indices)
    predicted_ratings = model.predict([user_array, anime_array], batch_size=256, verbose=0).flatten()
    top_indices = np.argsort(predicted_ratings)[-top_n:][::-1]
    recommended_indices = [unrated_indices[i] for i in top_indices]
    return [unique_animes[idx] for idx in recommended_indices]


# In[ ]:


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Extract test data
test_users = test_df['user_id'].values
test_animes = test_df['anime_id'].values
test_ratings = test_df['rating'].values


# In[ ]:


# --- NCF Predictions ---
# Step 1: Prepare user and anime indices for the entire test set
test_user_indices = np.array([user_id_to_index[user] for user in test_users])
test_anime_indices = np.array([anime_id_to_index[anime] for anime in test_animes])

# Step 2: Predict ratings in batches for all test samples
ncf_predictions = model.predict([test_user_indices, test_anime_indices], batch_size=256, verbose=1).flatten()


# In[ ]:


# --- SVD Predictions (if applicable) ---
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Extract test data
test_users = test_df['user_id'].values
test_animes = test_df['anime_id'].values
test_ratings = test_df['rating'].values

# Get indexer for users and animes
user_positions = predicted_ratings_df.index.get_indexer(test_users)
anime_positions = predicted_ratings_df.columns.get_indexer(test_animes)

# Find valid indices where both user and anime are present
valid_idx = (user_positions != -1) & (anime_positions != -1)

# Filter test data
test_users_valid = test_users[valid_idx]
test_animes_valid = test_animes[valid_idx]
test_ratings_valid = test_ratings[valid_idx]

# Get positions for valid entries
user_pos_valid = user_positions[valid_idx]
anime_pos_valid = anime_positions[valid_idx]

# Get predictions
svd_predictions = predicted_ratings_df.values[user_pos_valid, anime_pos_valid]

# Compute RMSE and MAE
rmse_svd = np.sqrt(mean_squared_error(test_ratings_valid, svd_predictions))
mae_svd = mean_absolute_error(test_ratings_valid, svd_predictions)

print(f"SVD RMSE: {rmse_svd:.2f}")
print(f"SVD MAE: {mae_svd:.2f}")


# In[ ]:


import numpy as np

# Convert rated_list to arrays
users, animes, ratings = zip(*rated_list)
users = np.array(users)
animes = np.array(animes)
ratings = np.array(ratings)

# Get predicted ratings
predicted = predicted_ratings_sgd[users, animes]

# Compute errors
errors = ratings - predicted

# Compute RMSE
rmse = np.sqrt(np.mean(errors**2))

# Compute MAE
mae = np.mean(np.abs(errors))


# In[ ]:


# NCF
rmse_ncf = np.sqrt(mean_squared_error(test_ratings, ncf_predictions))
mae_ncf = mean_absolute_error(test_ratings, ncf_predictions)


# In[ ]:


# Print results
print(f"NCF - RMSE: {rmse_ncf:.2f} MAE: {mae_ncf:.2f}")


# In[ ]:


print(f"Collaborative filtering - RMSE: {rmse:.2f},MAE: {mae:.2f}")


# In[ ]:




