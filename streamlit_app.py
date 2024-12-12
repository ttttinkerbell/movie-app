import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib
import json
import zipfile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from category_encoders import LeaveOneOutEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.svm import SVR

st.set_page_config(page_title="Movie Analysis System", layout="wide")

CREDENTIALS_FILE = "Users.json"

@st.cache_data
def load_data():
    try:
        zip_file_path = 'example.csv.zip'
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            csv_file_name = 'example.csv'
            with z.open(csv_file_name) as csv_file:
                df = pd.read_csv(csv_file)
    except Exception as e:
        try:
            df = pd.read_csv('example.csv')
        except Exception as e:
            st.error("Data file not found. Loading sample data instead.")
            np.random.seed(42)
            df = pd.DataFrame({
                'userId': np.random.randint(1, 1000, 1000),
                'movieId': np.random.randint(1, 100, 1000),
                'rating': np.random.uniform(1, 5, 1000),
                'title': [f'Movie {i} (2020)' for i in range(1000)], 
                'genres': np.random.choice(['Action', 'Comedy', 'Drama'], 1000)
            })
    return df

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_credentials():
    try:
        with open(CREDENTIALS_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_credentials(credentials):
    with open(CREDENTIALS_FILE, "w") as file:
        json.dump(credentials, file)

def login():
    st.subheader("Log In")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Log In"):
        credentials = load_credentials()
        hashed_password = hash_password(password)
        if username in credentials and credentials[username]["password"] == hashed_password:
            st.success(f"Welcome, {username}!")
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["userId"] = credentials[username]["id"]
            st.rerun()  
        else:
            st.error("Invalid username or password")
    return False

def generate_unique_user_id(existing_ids):
    new_id = np.random.randint(1000, 9999)
    while str(new_id) in existing_ids:
        new_id = np.random.randint(1000, 9999)  # Regenerate if ID already exists
    return str(new_id)

def signup():
    st.subheader("Sign Up")
    username = st.text_input("Choose a Username", key="signup_username")
    password = st.text_input("Choose a Password", type="password", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
    
    if st.button("Sign Up"):
        if not username or not password:
            st.error("Username and password are required.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            credentials = load_credentials()

            for user, data in credentials.items():
                if not isinstance(data, dict) or 'id' not in data:
                    credentials[user] = {
                        "id": generate_unique_user_id(set(credentials.keys())),
                        "password": data if isinstance(data, str) else hash_password(password)
                    }
            save_credentials(credentials)  # Save updated structure
            
            # Generate unique user ID
            existing_ids = {user_data['id'] for user_data in credentials.values() if isinstance(user_data, dict)}
            new_user_id = generate_unique_user_id(existing_ids)

            if username in credentials:
                st.error("Username already exists.")
            else:
                credentials[username] = {
                    "id": new_user_id,
                    "password": hash_password(password)
                }
                save_credentials(credentials)
                st.success(f"Sign up successful! Your User ID is {new_user_id}. Please log in.")

def find_movie_recommendations(df, user_input, k=5):
    if not user_input:
        return []
    
    matching_movies = df[df['title'].str.contains(user_input, case=False, na=False)]
    if matching_movies.empty:
        st.warning("No movies found matching your input.")
        return []

    st.subheader(f"Movies matching '{user_input}':")
    st.dataframe(matching_movies[['movieId', 'title', 'genres']].head(5))

    tfidf = TfidfVectorizer(stop_words='english')
    genres_and_titles = df['genres'].fillna('') + ' ' + df['title'].fillna('')
    tfidf_matrix = tfidf.fit_transform(genres_and_titles)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    movie_idx = matching_movies.index[0]
    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k + 1]

    return [(df.iloc[i[0]]['movieId'], df.iloc[i[0]]['title'], 
             df.iloc[i[0]]['genres'], i[1]) for i in sim_scores]

def get_movie_recommendations(df, user_input, k=5):
    if not user_input:
        return []
    
    matching_movies = df[df['title'].str.contains(user_input, case=False, na=False)]
    if matching_movies.empty:
        st.warning("No movies found matching your input.")
        return []

    st.subheader(f"Movies matching '{user_input}':")
    st.dataframe(matching_movies[['movieId', 'title', 'genres']].head(5))

    tfidf = TfidfVectorizer(stop_words='english')
    genres_and_titles = df['genres'].fillna('') + ' ' + df['title'].fillna('')
    tfidf_matrix = tfidf.fit_transform(genres_and_titles)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    movie_idx = matching_movies.index[0]
    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k + 1]

    return [(df.iloc[i[0]]['movieId'], df.iloc[i[0]]['title'], 
             df.iloc[i[0]]['genres'], i[1]) for i in sim_scores]


def train_models(df):
    df_sampled = df.sample(n=min(len(df), 10000), random_state=42)
    
    df_sampled['release_year'] = pd.to_numeric(
        df_sampled['title'].str.extract(r'\((\d{4})\)', expand=False),
        errors='coerce'
    )
    df_sampled['release_year'].fillna(df_sampled['release_year'].median(), inplace=True)
    
    df_sampled['genres'] = df_sampled['genres'].fillna('Unknown').str.split('|')
    genres_exploded = df_sampled['genres'].explode()
    top_genres = genres_exploded.value_counts().nlargest(10).index.tolist()
    
    for genre in top_genres:
        df_sampled[f'genre_{genre}'] = df_sampled['genres'].apply(
            lambda x: int(genre in x if isinstance(x, list) else False)
        )
    
    feature_columns = ['release_year'] + [f'genre_{genre}' for genre in top_genres]
    X = df_sampled[feature_columns].copy()
    y = df_sampled['rating'].copy()
    
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    df_sampled['userId_encoded'] = user_encoder.fit_transform(df_sampled['userId'])
    df_sampled['movieId_encoded'] = movie_encoder.fit_transform(df_sampled['movieId'])

    X = pd.concat([df_sampled[['userId_encoded', 'movieId_encoded']], X], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    knn = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.2)
    svr.fit(X_train_scaled, y_train)
    y_pred_svr = svr.predict(X_test_scaled)
    
    metrics = {
        'rf_metrics': {
            'mse': mean_squared_error(y_test, y_pred_rf),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            'mae': mean_absolute_error(y_test, y_pred_rf),
            'r2': r2_score(y_test, y_pred_rf)
        },
        'knn_metrics': {
            'mse': mean_squared_error(y_test, y_pred_knn),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_knn)),
            'mae': mean_absolute_error(y_test, y_pred_knn),
            'r2': r2_score(y_test, y_pred_knn)
        },
        'svm_metrics': {
            'mse': mean_squared_error(y_test, y_pred_svr),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_svr)),
            'mae': mean_absolute_error(y_test, y_pred_svr),
            'r2': r2_score(y_test, y_pred_svr)
        }
    }
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        **metrics,
        'df_sampled': df_sampled,
        'results': {
            'rf': {'actual': y_test, 'predicted': y_pred_rf},
            'knn': {'actual': y_test, 'predicted': y_pred_knn},
            'svm': {'actual': y_test, 'predicted': y_pred_svr}
        },
        'feature_importance': feature_importance
    }

def create_comparison_visualizations(results):
    metrics_df = pd.DataFrame({
        'Model': ['Random Forest', 'KNN', 'SVM'],
        'MSE': [results['rf_metrics']['mse'], results['knn_metrics']['mse'], results['svm_metrics']['mse']],
        'RMSE': [results['rf_metrics']['rmse'], results['knn_metrics']['rmse'], results['svm_metrics']['rmse']],
        'MAE': [results['rf_metrics']['mae'], results['knn_metrics']['mae'], results['svm_metrics']['mae']],
        'R²': [results['rf_metrics']['r2'], results['knn_metrics']['r2'], results['svm_metrics']['r2']]
    })

    fig1 = make_subplots(rows=2, cols=2, subplot_titles=('MSE', 'RMSE', 'MAE', 'R²'))
    metrics = ['MSE', 'RMSE', 'MAE', 'R²']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for metric, pos in zip(metrics, positions):
        fig1.add_trace(
            go.Bar(x=metrics_df['Model'], y=metrics_df[metric], name=metric),
            row=pos[0], col=pos[1]
        )
    
    fig1.update_layout(height=800, title_text='Model Performance Comparison', showlegend=False)
    
    fig2 = make_subplots(rows=1, cols=3, subplot_titles=('Random Forest', 'KNN', 'SVM'))
    models = ['rf', 'knn', 'svm']
    
    for i, model in enumerate(models, 1):
        fig2.add_trace(
            go.Scatter(
                x=results['results'][model]['actual'],
                y=results['results'][model]['predicted'],
                mode='markers',
                name=model.upper(),
                opacity=0.6
            ),
            row=1, col=i
        )
    
    fig2.update_layout(height=400, title_text='Actual vs Predicted Values')
    
    fig3 = make_subplots(rows=1, cols=3, subplot_titles=(
        'Random Forest Residuals',
        'KNN Residuals',
        'SVM Residuals'
    ))
    
    for i, model in enumerate(models, 1):
        residuals = results['results'][model]['actual'] - results['results'][model]['predicted']
        fig3.add_trace(go.Histogram(x=residuals, name=model.upper()), row=1, col=i)
    
    fig3.update_layout(height=400, title_text='Residuals Distribution')
    
    return fig1, fig2, fig3, metrics_df

RATINGS_FILE = "ratings.json"

def load_ratings():
    try:
        with open(RATINGS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_ratings(new_ratings):
    try:
        existing_ratings = load_ratings()
        transformed_existing_ratings = [
            {
                "userId": r.get("userId", r.get("userId")),
                "title": r.get("title", r.get("title")),
                "rating": r["rating"]
            }
            for r in existing_ratings
        ]
        transformed_new_ratings = [
            {
                "userId": r["userId"],
                "title": r["title"],
                "rating": r["rating"]
            }
            for r in new_ratings
        ]
        combined_ratings = transformed_existing_ratings + transformed_new_ratings
        unique_ratings = {f"{r['userId']}_{r['title']}": r for r in combined_ratings}.values()
        
        with open(RATINGS_FILE, "w") as f:
            json.dump(list(unique_ratings), f)

        # Update session state with the new ratings
        for new_rating in new_ratings:
            update_session_data(new_rating)

    except Exception as e:
        st.error(f"Error saving ratings: {str(e)}")

def merge_ratings_with_dataset(original_df, new_ratings):
    if not isinstance(new_ratings, pd.DataFrame):
        new_ratings_df = pd.DataFrame(new_ratings)
    else:
        new_ratings_df = new_ratings.copy()

    new_ratings_df = new_ratings_df.rename(columns={
        "userId": "userId",
        "title": "title",
        "rating": "rating"
    })

    new_ratings_df['userId'] = new_ratings_df['userId'].astype(int)
    matched_df = pd.merge(
        new_ratings_df,
        original_df[['title', 'movieId', 'genres']].drop_duplicates(subset=['title']),
        how='left',
        on='title'
    )

    combined_df = original_df.copy()
    for _, new_rating in matched_df.iterrows():
        combined_df = combined_df[
            ~((combined_df['userId'] == new_rating['userId']) & 
              (combined_df['title'] == new_rating['title']))
        ]

    combined_df = pd.concat([combined_df, matched_df], ignore_index=True)
    return combined_df


def update_session_data(new_rating):
    if "df" in st.session_state:
        st.session_state.df = merge_ratings_with_dataset(
            st.session_state.df, 
            [new_rating]
        )

def main():
    if "df" not in st.session_state:
        original_df = load_data()
        saved_ratings = pd.DataFrame(load_ratings())
        if not saved_ratings.empty:
            st.session_state.df = merge_ratings_with_dataset(original_df, saved_ratings)
        else:
            st.session_state.df = original_df

    st.title("🎬 Movie Analysis and Recommendation System")
    
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    
    st.sidebar.title("Navigation")
    if not st.session_state["logged_in"]:
        auth_option = st.sidebar.selectbox("Choose an option", ["Log In", "Sign Up"])
        if auth_option == "Log In":
            login()
        else:
            signup()
        return
    
    st.sidebar.success(f"Logged in as {st.session_state['username']}")
    st.sidebar.info(f"User ID: {st.session_state['userId']}")
    if st.sidebar.button("Log Out"):
        st.session_state["logged_in"] = False
        st.rerun()  
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Find Movie", "Unrated Movie Recommendations", "Similar Preference Recommendations", "Model Analysis", "User Ratings"])

    with tab1:
        st.header("Find Movie")

        user_input = st.text_input("Enter a movie title:", value="", key="movie_title_tab1")
        if st.button("Search", key="search_tab1"):
            recommendations = find_movie_recommendations(st.session_state.df, user_input)
            if recommendations:
                st.subheader("Movies:")
                for movie_idx, title, genres, similarity in recommendations:
                    st.write(f"🎥 {title} ({genres}) - Similarity: {similarity:.2f}")
            else:
                st.warning("Please try another title.")
        
        st.subheader("All the movies")
        movie_list = st.session_state.df['title'].drop_duplicates().sort_values().tolist()
        selected_movie = st.selectbox("Here are the movies we have recorded:", options=movie_list, key="movie_dropdown")
        
        # Movie rating
        st.subheader("Rate Movie (1 to 5 stars)")
    
        if selected_movie:
            st.write(f"🎥 You selected: {selected_movie}")
            rating = st.slider(f"Rate '{selected_movie}':", min_value=1, max_value=5, step=1, key="movie_rating")
            if st.button("Submit Rating", key="submit_rating"):
                new_rating = {
                    "userId": st.session_state["userId"],
                    "title": selected_movie,
                    "rating": rating
                }
                save_ratings([new_rating])

                updated_ratings = pd.DataFrame(load_ratings()).rename(columns={
                    "userId": "userId",
                    "title": "title"
                })

                if not updated_ratings.empty:
                    merged_df = pd.merge(
                        updated_ratings,
                        st.session_state.df[['title', 'movieId', 'genres']].drop_duplicates(),
                        how='left',
                        on='title'
                    )
                    st.session_state.df = pd.concat([st.session_state.df, merged_df], ignore_index=True)
        
                st.success(f"Rating saved! You rated '{selected_movie}' with {rating} stars.")
                st.rerun()

    with tab2:
        st.header("Unrated Movie Recommendations")
        st.write("We will recommend movies to you based on your previous rating preferences.")

        user_id = st.number_input("Enter Your User ID:", min_value=1, value=int(st.session_state["userId"]), key="tab2_user_id")
        user_ratings = st.session_state.df[st.session_state.df['userId'] == user_id]

        if user_ratings.empty:
            st.warning("You haven't rated any movies yet. Please rate some movies to get personalized recommendations.")
        else:
            ratings_matrix = st.session_state.df.pivot_table(
                index='userId',
                columns='title',
                values='rating',
                fill_value=0
            )

            user_similarity = cosine_similarity(ratings_matrix)
            user_similarity_df = pd.DataFrame(
                user_similarity,
                index=ratings_matrix.index,
                columns=ratings_matrix.index
            )

            similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]  # Top 5 similar users
            similar_user_ids = similar_users.index

            rated_movies = user_ratings['title'].tolist()
            unrated_movies = st.session_state.df[~st.session_state.df['title'].isin(rated_movies)]

            predictions = []
            for movie in unrated_movies['title'].unique():
                movie_ratings = st.session_state.df[st.session_state.df['title'] == movie]
                relevant_ratings = movie_ratings[movie_ratings['userId'].isin(similar_user_ids)]

                if not relevant_ratings.empty:
                    weighted_rating = 0
                    total_similarity = 0
                    for _, row in relevant_ratings.iterrows():
                        similarity = similar_users.get(row['userId'], 0)
                        weighted_rating += row['rating'] * similarity
                        total_similarity += similarity

                    if total_similarity > 0:
                        predicted_rating = weighted_rating / total_similarity
                        predictions.append({
                            'title': movie,
                            'genres': movie_ratings.iloc[0]['genres'],
                            'predicted_rating': predicted_rating
                        })

            if predictions:
                recommendations_df = pd.DataFrame(predictions).sort_values(by='predicted_rating', ascending=False)

                st.subheader("Recommended Movies Based on Your Ratings")
                top_recommendations = recommendations_df.head(5)
                st.dataframe(top_recommendations[['title', 'genres', 'predicted_rating']], use_container_width=True)
            else:
                st.info("No recommendations available. Try rating more movies to get better suggestions.")

    with tab3:
        st.header("Similar Preference Recommendations")
        st.write("We will recommend users who have the same preferences as you, and movies they like.")

        userId = st.number_input("Enter Your User ID:", min_value=1, value=int(st.session_state["userId"]))
        ratings_matrix = st.session_state.df.pivot_table(index='userId', columns='title', values='rating')
        
        if userId not in ratings_matrix.index:
            st.warning("You haven't rated any movies yet. Please rate some movies to get recommendations.")
        else:
            ratings_matrix_filled = ratings_matrix.fillna(0)
            user_similarities = cosine_similarity(ratings_matrix_filled)
            similarity_df = pd.DataFrame(user_similarities, index=ratings_matrix.index, columns=ratings_matrix.index)
            
            similar_users = similarity_df[userId].sort_values(ascending=False).iloc[1:6]
            st.subheader("Top 5 Users with Similar Preferences")
            st.dataframe(similar_users.reset_index().rename(columns={userId: "Similarity"}), use_container_width=True)
            
            similar_user_ids = similar_users.index
            similar_user_ratings = ratings_matrix.loc[similar_user_ids]
            
            recommended_movies = similar_user_ratings.mean(axis=0).sort_values(ascending=False)
            user_rated_movies = ratings_matrix.loc[userId].dropna().index
            recommended_movies = recommended_movies.drop(user_rated_movies, errors='ignore')
            
            st.subheader("Movies Recommended Based on Similar Users")
            st.write("These movies are rated highly by users with similar preferences:")
            top_recommendations = recommended_movies.head(5).reset_index()
            top_recommendations.columns = ['Movie Title', 'Average Rating']
            st.dataframe(top_recommendations, use_container_width=True)

    with tab4:
        st.header("Model Performance Analysis")
        if st.button("Run Analysis"):
            with st.spinner("Training models and generating analysis..."):
                try:
                    results = train_models(st.session_state.df)
                    fig1, fig2, fig3, metrics_df = create_comparison_visualizations(results)
                    
                    st.subheader("Model Metrics Comparison")
                    st.dataframe(metrics_df)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    st.subheader("Feature Importance (Random Forest)")
                    fig4 = px.bar(
                        results['feature_importance'],
                        x='feature',
                        y='importance',
                        title='Feature Importance Analysis'
                    )
                    st.plotly_chart(fig4, use_container_width=True)
                    
                    st.subheader("Rating Distribution Analysis")
                    fig5 = px.histogram(
                        results['df_sampled'],
                        x='rating',
                        title='Overall Rating Distribution',
                        nbins=20,
                        color_discrete_sequence=['#1f77b4']
                    )
                    st.plotly_chart(fig5, use_container_width=True)
                    
                    avg_rating_by_year = results['df_sampled'].groupby('release_year')['rating'].agg([
                        'mean',
                        'count'
                    ]).reset_index()
                    avg_rating_by_year = avg_rating_by_year[avg_rating_by_year['count'] > 10]  
                    fig6 = px.line(
                        avg_rating_by_year,
                        x='release_year',
                        y='mean',
                        title='Average Rating by Release Year',
                        labels={'mean': 'Average Rating', 'release_year': 'Release Year'}
                    )
                    st.plotly_chart(fig6, use_container_width=True)
                    
                    genre_cols = [col for col in results['df_sampled'].columns if col.startswith('genre_')]
                    genre_ratings = []
                    
                    for genre in genre_cols:
                        genre_name = genre.replace('genre_', '')
                        mean_rating = results['df_sampled'][results['df_sampled'][genre] == 1]['rating'].mean()
                        count = results['df_sampled'][results['df_sampled'][genre] == 1]['rating'].count()
                        genre_ratings.append({
                            'Genre': genre_name,
                            'Average Rating': mean_rating,
                            'Count': count
                        })
                    
                    genre_ratings_df = pd.DataFrame(genre_ratings)
                    fig7 = px.bar(
                        genre_ratings_df,
                        x='Genre',
                        y='Average Rating',
                        title='Average Rating by Genre',
                        text='Count'
                    )
                    fig7.update_traces(texttemplate='%{text} ratings')
                    st.plotly_chart(fig7, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
    
    with tab5:
        st.header("User Rating Analysis")
        userId = st.number_input("Enter User ID", min_value=1, value=int(st.session_state["userId"]))
        
        user_ratings = st.session_state.df[st.session_state.df['userId'] == userId]
        
        if not user_ratings.empty:
            st.subheader("Your Movie Ratings")
            st.dataframe(
                user_ratings[['title', 'rating', 'genres']].sort_values('rating', ascending=False),
                use_container_width=True
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig8 = px.histogram(
                    user_ratings,
                    x='rating',
                    title='Your Rating Distribution',
                    nbins=10,
                    color_discrete_sequence=['#2ecc71']
                )
                st.plotly_chart(fig8, use_container_width=True)
            
            with col2:
                genres = user_ratings['genres'].str.split('|', expand=True).stack()
                genre_ratings = user_ratings.loc[genres.index.get_level_values(0)].copy()
                genre_ratings['genre'] = genres.values
                genre_avg = genre_ratings.groupby('genre')['rating'].agg([
                    'mean',
                    'count'
                ]).reset_index()
                
                fig9 = px.bar(
                    genre_avg,
                    x='genre',
                    y='mean',
                    title='Your Average Rating by Genre',
                    text='count',
                    labels={'mean': 'Average Rating', 'genre': 'Genre'}
                )
                fig9.update_traces(texttemplate='%{text} ratings')
                st.plotly_chart(fig9, use_container_width=True)
            
            user_ratings['release_year'] = user_ratings['title'].str.extract(r'\((\d{4})\)')
            user_ratings['release_year'] = pd.to_numeric(user_ratings['release_year'], errors='coerce')
            yearly_ratings = user_ratings.groupby('release_year')['rating'].agg([
                'mean',
                'count'
            ]).reset_index()
            yearly_ratings = yearly_ratings[yearly_ratings['count'] > 2] 
            
            fig10 = px.line(
                yearly_ratings,
                x='release_year',
                y='mean',
                title='Your Rating Trend Over Movie Release Years',
                labels={'mean': 'Average Rating', 'release_year': 'Release Year'}
            )
            st.plotly_chart(fig10, use_container_width=True)
            
            st.subheader("Your Ratings vs Overall Average")
            overall_avg = st.session_state.df.groupby('movieId')['rating'].mean().reset_index()
            user_vs_avg = user_ratings.merge(
                overall_avg,
                on='movieId',
                suffixes=('_user', '_overall')
            )
            
            fig11 = px.scatter(
                user_vs_avg,
                x='rating_overall',
                y='rating_user',
                title='Your Ratings vs Overall Average',
                labels={
                    'rating_overall': 'Overall Average Rating',
                    'rating_user': 'Your Rating'
                }
            )

            fig11.add_shape(
                type='line',
                x0=user_vs_avg['rating_overall'].min(),
                y0=user_vs_avg['rating_overall'].min(),
                x1=user_vs_avg['rating_overall'].max(),
                y1=user_vs_avg['rating_overall'].max(),
                line=dict(color='red', dash='dash')
            )
            
            st.plotly_chart(fig11, use_container_width=True)
            
            st.subheader("Rating Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Ratings", len(user_ratings))
            with col2:
                st.metric("Average Rating", f"{user_ratings['rating'].mean():.2f}")
            with col3:
                st.metric("Unique Genres", len(genre_ratings['genre'].unique()))
            with col4:
                if not yearly_ratings['release_year'].isna().all():
                    valid_years = yearly_ratings['release_year'].dropna()
                    st.metric("Years Span", f"{int(valid_years.max() - valid_years.min())}")
                else:
                    st.metric("Years Span", "No valid years")
        else:
            st.info("No ratings found for this user ID.")
                
if __name__ == "__main__":
    main()
