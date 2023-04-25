#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[14]:


# Load the dataset into a pandas DataFrame
df = pd.read_csv("Datafiniti_Hotel_Reviews_Jun19.csv")

# Check for missing values
df.isnull().sum()

#Remove duplicates
df.drop_duplicates(inplace=True)

# Remove irrelevant columns
df.drop(['id', 'dateAdded','dateUpdated','reviews.dateAdded','longitude','province','postalCode','reviews.date', 'address','primaryCategories','country','keys', 'latitude','reviews.dateSeen','reviews.sourceURLs', 'reviews.title', 'reviews.userProvince', 'reviews.username', 'sourceURLs' , 'websites' ], axis=1, inplace=True)

# Rename the 'A' column to 'X'
df = df.rename(columns={'reviews.rating': 'rating', 'reviews.text':'reviews' })

# Print the updated column names
print(df.columns)

# Convert data types
df['rating'] = df['rating'].astype(float)

# Define the stop words
stop_words = set(stopwords.words('english'))

# Handle text data
df['reviews'] = df['reviews'].str.lower()
df['reviews'] = df['reviews'].apply(word_tokenize)
df['reviews'] = df['reviews'].apply(lambda x: [item for item in x if item not in stop_words])
df['reviews'] = df['reviews'].apply(lambda x: ' '.join(x))

# Create a new column for review sentiment based on rating
df['review_sentiment'] = df['rating'].apply(lambda x: 'Positive' if x > 3 else ('Neutral' if x == 3 else 'Negative'))


# In[15]:


df


# In[16]:



#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['reviews'], df['review_sentiment'], test_size=0.2, random_state=42)

#Tokenize, clean, and transform the training and testing data
stop_words = set(stopwords.words('english'))
vect = CountVectorizer(stop_words='english')
X_train_vect = vect.fit_transform(X_train.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words])))
X_test_vect = vect.transform(X_test.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words])))

#Train the SVM model
svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(X_train_vect, y_train)

# Predict the sentiment of the test set
y_pred = svm.predict(X_test_vect)

from sklearn.metrics import accuracy_score, precision_score, recall_score

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

acc_percentage = round(accuracy * 100, 1)
percentage_str1 = str(acc_percentage) + '%'

pre_percentage = round(precision * 100, 1)
percentage_str2 = str(pre_percentage) + '%'

re_percentage = round(recall * 100, 1)
percentage_str3 = str(re_percentage) + '%'


# Print the results
print("The Accuracy of the SVM model is:", percentage_str1)
print("The Precision of the SVM model is:",percentage_str2 )
print("The Recall of the SVM model is:", percentage_str3)


from sklearn.metrics import f1_score
# Calculate the F1 score
f1score = f1_score(y_test, y_pred, average='weighted')
f1_percentage = round(f1score * 100, 1)
percentage_str4 = str(f1_percentage) + '%'
print("The F1 score of the SVM model is:", percentage_str4)



# In[17]:


#Predict the sentiment of a statement entered by a user
statement = input("Enter a statement to predict its sentiment: ")
statement = statement.lower()
statement = word_tokenize(statement)
statement = [item for item in statement if item not in stop_words]
statement = ' '.join(statement)
statement_vect = vect.transform([statement])
prediction = svm.predict(statement_vect)
print("The predicted sentiment of the statement is:", prediction[0])


# In[18]:



from sklearn.naive_bayes import MultinomialNB

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['reviews'], df['review_sentiment'], random_state=0)

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Fit the vectorizer to the training data
X_train_vect = vectorizer.fit_transform(X_train)

# Train a Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X_train_vect, y_train)

# Vectorize the testing data
X_test_vect = vectorizer.transform(X_test)

# Calculate the accuracy of the model on the testing data
y_pred = nb.predict(X_test_vect)

# Calculate accuracy, precision, and recall
accuracy2 = accuracy_score(y_test, y_pred)
precision2 = precision_score(y_test, y_pred, average='weighted')
recall2 = recall_score(y_test, y_pred, average='weighted')

acc_percentage = round(accuracy2 * 100, 1)
per_str1 = str(acc_percentage) + '%'

pre_percentage = round(precision2 * 100, 1)
per_str2 = str(pre_percentage) + '%'

re_percentage = round(recall2 * 100, 1)
per_str3 = str(re_percentage) + '%'


# Print the results
print("The Accuracy of the NB model is:", per_str1)
print("The Precision of the NB model is:",per_str2 )
print("The Recall of the NB model is:", per_str3)


from sklearn.metrics import f1_score
# Calculate the F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
f1_per = round(f1 * 100, 1)
per_str4 = str(f1_per) + '%'
print("The F1 score of the NB model is:", per_str4)




# In[19]:


# Predict the sentiment of a statement entered by a user
statement2 = input("Enter a statement to predict its sentiment: ")
statement_vect2 = vectorizer.transform([statement2])
predicted_sentiment = nb.predict(statement_vect2)
print("The predicted sentiment of the statement is:", predicted_sentiment)


# In[20]:


import matplotlib.pyplot as plt
# Group the reviews by sentiment and count the number of reviews in each category
sentiment_counts = df.groupby('review_sentiment').size()
print(sentiment_counts)
# Create a bar chart to show the distribution of review sentiments
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.title('Distribution of Review Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.show()


# In[21]:


df['rating'].value_counts() # Now we can determine which ones are positive and negative reviews


# In[22]:


# Group the reviews by sentiment and count the number of reviews in each category
rating_counts = df.groupby('rating').size()
print(rating_counts)
# Create a bar chart to show the distribution of review sentiments
plt.bar(rating_counts.index, rating_counts.values)
plt.title('Distribution of Rating counts')
plt.xlabel('Rating')
plt.ylabel('Number of Ratings')
plt.show()

