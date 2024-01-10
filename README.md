INDIVIDUAL PROYECT

This project involves creating an API that utilizes a recommendation model for Steam, a multinational video game platform, based on Machine Learning. The goal is to build a video game recommendation system for users. The API provides an intuitive interface for users to obtain information for the recommendation system and data on genres or specific dates.

Tools Used:
-Render
-FastAPI
-Python
-Scikit-Learn
-Step-by-Step:
-Pandas
-Matplotlib
-Numpy
-Seaborn
-Wordcloud
-NLTK
-Uvicorn

ETL
Conducted an ETL (Extraction, Transformation, and Loading) process where data was extracted from various sources, transformed according to the project's needs, and loaded into a final destination for analysis and further use. The primary tools used were Python, Pandas, Scikit-Learn, and FastAPI.

API Deployment
Created an API using the FastAPI module in Python, implementing five functions that can be queried:

PlayTimeGenre: Returns the year with the most played hours for a given genre. 
UserForGenre: Returns the user with the most accumulated hours played for the given genre and a list of accumulated hours played per year.
UsersRecommend: Returns the top 3 MOST recommended games by users for the given year. (reviews.recommend = True and positive/neutral comments) 
UsersWorstDeveloper(year: int): Returns the top 3 developers with the LEAST recommended games by users for the given year.
sentiment_analysis: According to the release year, returns a list with the count of user review records categorized with sentiment analysis. 

EDA (Exploratory Data Analysis)
Conducted an exploratory data analysis process where data was explored and analyzed thoroughly to gain insights, identify patterns, trends, and relationships, with the aim of making informed decisions based on the obtained information. The tools used were Numpy, Pandas, Matplotlib, Seaborn, Wordcloud, NLTK.

Machine Learning Model
Developed a machine learning model to generate game recommendations using algorithms and techniques such as cosine similarity and scikit-learn. The goal was to provide personalized and accurate recommendations based on the tastes and preferences of each user. If it is an item-item recommendation system:

recomendacion_juego (game recommendation): By entering the product id ('id'), we should receive a list of 5 games recommended similar to the entered one.
Example usage: 70
If it is a user-item recommendation system:
recomendacion_usuario (user recommendation): By entering the user id ('user_id'), we should receive a list of 5 games recommended for that user.
Example usage: 76561198030567998
The tool used was Scikit-Learn with libraries: TfidfVectorizer, linear_kernel, cosine_similarity. These functionalities are also queryable in the API.
