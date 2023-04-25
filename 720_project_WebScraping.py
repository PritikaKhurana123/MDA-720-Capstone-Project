#!/usr/bin/env python
# coding: utf-8

# In[47]:


from yelpapi import YelpAPI
import pandas as pd

# set up the Yelp API client with your API key
API_KEY = "LVLsPKz4MN_AYielbfiVleguaihF5S8fq_gPlo1yqpm0qJEAxCGRNj62JMOW9lE6d-KfGJvEpAshQQXImiO2Gwm-B27T___qxXNGhM1NFV74_FHPnBlEv2ThCqE0ZHYx"
yelp_api = YelpAPI(API_KEY)

# use the Yelp API to search for hotels in Manhattan with lowest ratings first
response = yelp_api.search_query(term="hotels", location="Manhattan, NY", sort_by="rating", limit=50, offset=50)

# create an empty list to store the hotel data
hotel_data = []

# loop through the hotels and extract the data
for business in response["businesses"]:
    name = business["name"]
    rating = business["rating"]
    review_count = business["review_count"]
    address = ", ".join(business["location"]["display_address"])
    
    # get up to three reviews for each hotel
    review_response = yelp_api.reviews_query(id=business["id"], limit=3)
    reviews = [review["text"] for review in review_response["reviews"]]
    
    hotel_data.append([name, rating, review_count, address, reviews])

# create a pandas dataframe with the hotel data
df = pd.DataFrame(hotel_data, columns=["Hotel Name", "Rating", "Review Count", "Address", "Reviews"])

# print the dataframe
print(df)


# In[3]:


from textblob import TextBlob

# define a function to perform sentiment analysis and return a sentiment label
def get_sentiment(review):
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# loop through the rows of the data frame and print the reviews with their sentiment labels
for index, row in df.iterrows():
    name = row["Hotel Name"]
    address = row["Address"]
    rating = row["Rating"]
    print(f"        ->{name} ({rating} stars):")
    print(f"\tAddress: {address}")
    for review in row["Reviews"]:
        sentiment = get_sentiment(review)
        print(f"\tReview: {review}")
        print(f"\tSentiment: {sentiment}\n")


# In[4]:


df['Hotel Name'].count()


# In[5]:


df['Rating'].unique()


# In[6]:


from nltk.corpus import stopwords
from collections import Counter
import string

# combine all the reviews into a single string
reviews_string = " ".join(df["Reviews"].sum())

# convert the string to lowercase and remove any punctuation
reviews_string = reviews_string.lower().translate(str.maketrans('', '', string.punctuation))

# split the string into individual words
reviews_words = reviews_string.split()

# remove any stop words
stop_words = set(stopwords.words('english'))
reviews_words = [word for word in reviews_words if word not in stop_words]

# count the frequency of each word
word_counts = Counter(reviews_words)

# sort the words by frequency and print out the most common ones
most_common_words = word_counts.most_common(10)
print("The 10 most common words in the reviews are:")
for word, count in most_common_words:
    print(f"{word}: {count}")


# In[ ]:


"""
This code will generate a word cloud visualization of the hotel reviews. The larger the word appears in the cloud, the more frequently it appears in the reviews. You can adjust the size and appearance of the word cloud by modifying the parameters of the WordCloud function.






"""


# In[29]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# join all the hotel reviews into a single string
reviews_text = " ".join(review for review in df["Reviews"].apply(lambda x: " ".join(x)))

# create the word cloud
wordcloud = WordCloud(width=800, height=400, max_font_size=100, background_color="white").generate(reviews_text)

# display the word cloud
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[21]:


#!pip install prettytable
from prettytable import PrettyTable

# define a set of positive words
positive_words = set(nltk.corpus.opinion_lexicon.positive())

# count the frequency of each positive word
positive_counts = {}
for word in reviews_words:
    if word in positive_words:
        if word in positive_counts:
            positive_counts[word] += 1
        else:
            positive_counts[word] = 1

# sort the positive words by frequency and create a table with the results
table = PrettyTable()
table.field_names = ["Positive Word", "Count"]
for word in sorted(positive_counts, key=positive_counts.get, reverse=True):
    table.add_row([word, positive_counts[word]])

# print the table
print(table)


# In[26]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# convert the positive words to a string
positive_words_string = ' '.join(sorted_positive_words)

# create a word cloud
wordcloud = WordCloud(background_color='white', width=800, height=400).generate(positive_words_string)

# plot the word cloud
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[24]:


# define a set of negative words
negative_words = set(nltk.corpus.opinion_lexicon.negative())

# count the frequency of each negative word
negative_counts = {}
for word in reviews_words:
    if word in negative_words:
        if word in negative_counts:
            negative_counts[word] += 1
        else:
            negative_counts[word] = 1

# sort the positive words by frequency and create a table with the results
table = PrettyTable()
table.field_names = ["Negative Word", "Count"]
for word in sorted(negative_counts, key=negative_counts.get, reverse=True):
    table.add_row([word, negative_counts[word]])

# print the table
print(table)


# In[25]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# convert the negative word counts to a string
negative_words_string = " ".join(sorted(negative_counts, key=negative_counts.get, reverse=True))

# create the word cloud
wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = set(), min_font_size = 10).generate(negative_words_string)

# plot the word cloud
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


# In[ ]:


"""This code will create a map of Manhattan with markers for each hotel, displaying the hotel name, rating, and address when clicked. We can use this map to visually inspect the geographic distribution of hotels in Manhattan and identify any areas with a high concentration of hotels."""


# In[61]:


from yelpapi import YelpAPI
import pandas as pd
import folium

# set up the Yelp API client with your API key
API_KEY = "LVLsPKz4MN_AYielbfiVleguaihF5S8fq_gPlo1yqpm0qJEAxCGRNj62JMOW9lE6d-KfGJvEpAshQQXImiO2Gwm-B27T___qxXNGhM1NFV74_FHPnBlEv2ThCqE0ZHYx"
yelp_api = YelpAPI(API_KEY)

# use the Yelp API to search for hotels in Manhattan with lowest ratings first
response = yelp_api.search_query(term="hotels", location="Manhattan, NY", sort_by="rating", limit=50, offset=50)

# create an empty list to store the hotel data
hotel_data = []

# loop through the hotels and extract the data
for business in response["businesses"]:
    name = business["name"]
    rating = business["rating"]
    review_count = business["review_count"]
    address = ", ".join(business["location"]["display_address"])
    latitude = business["coordinates"]["latitude"] # add latitude information
    longitude = business["coordinates"]["longitude"] # add longitude information
    
    # get up to three reviews for each hotel
    review_response = yelp_api.reviews_query(id=business["id"], limit=3)
    reviews = [review["text"] for review in review_response["reviews"]]
    
    hotel_data.append([name, rating, review_count, address, reviews, latitude, longitude]) # add latitude and longitude information to the hotel data

# create a pandas dataframe with the hotel data
df = pd.DataFrame(hotel_data, columns=["Hotel Name", "Rating", "Review Count", "Address", "Reviews", "Latitude", "Longitude"])

# set the center of the map to Manhattan
manhattan_coords = [40.7831, -73.9712]

# create a map object
map = folium.Map(location=manhattan_coords, zoom_start=12)

# loop through the hotels and add a marker for each one
for index, row in df.iterrows():
    name = row["Hotel Name"]
    rating = row["Rating"]
    address = row["Address"]
    reviews = row["Reviews"]
    lat = row["Latitude"]
    lon = row["Longitude"]
    
    # create a popup with the hotel name, rating, and address
    popup_text = f"<b>{name}</b><br>Rating: {rating}<br>Address: {address}"
    
    # add the marker to the map
    folium.Marker(location=[lat, lon], popup=popup_text).add_to(map)

# display the map
map


# In[48]:


import matplotlib.pyplot as plt

# create a horizontal bar chart of the hotels and their ratings
fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(df["Hotel Name"], df["Rating"])

# set the x-axis label
ax.set_xlabel("Rating")

# set the title of the chart
ax.set_title("Hotel Ratings in Manhattan")

# show the chart
plt.show()


# In[34]:


import matplotlib.pyplot as plt

# create a bar chart of hotel ratings
df["Rating"].value_counts().sort_index().plot(kind="bar")

# set chart title and axis labels
plt.title("Distribution of Hotel Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")

# show the chart
plt.show()


# In[35]:


import matplotlib.pyplot as plt

# plot rating vs. review count
plt.scatter(df["Rating"], df["Review Count"])

# set plot title and labels
plt.title("Hotel Rating vs. Review Count")
plt.xlabel("Rating")
plt.ylabel("Review Count")

# display the plot
plt.show()


# In[37]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# combine all the reviews into a single string
reviews_text = ' '.join([review for reviews in df['Reviews'].tolist() for review in reviews])

# create the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reviews_text)

# plot the word cloud
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[42]:


# create a new DataFrame with just the review count and rating columns
review_count_rating = df[["Review Count", "Rating"]]


# In[45]:


# calculate summary statistics for the review count and rating variables
print(review_count_rating.describe())


# In[44]:


# create a scatter plot of the review count and rating variables
review_count_rating.plot.scatter(x="Review Count", y="Rating")


# In[59]:



from nltk.sentiment.vader import SentimentIntensityAnalyzer

#nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()
# set up the Yelp API client with your API key
API_KEY = "LVLsPKz4MN_AYielbfiVleguaihF5S8fq_gPlo1yqpm0qJEAxCGRNj62JMOW9lE6d-KfGJvEpAshQQXImiO2Gwm-B27T___qxXNGhM1NFV74_FHPnBlEv2ThCqE0ZHYx"
yelp_api = YelpAPI(API_KEY)

# use the Yelp API to search for hotels in Manhattan with lowest ratings first
response = yelp_api.search_query(term="hotels", location="Manhattan, NY", sort_by="rating", limit=50, offset=50)

# create an empty list to store the hotel data
hotel_data = []

# initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# loop through the hotels and extract the data
for business in response["businesses"]:
    name = business["name"]
    rating = business["rating"]
    review_count = business["review_count"]
    address = ", ".join(business["location"]["display_address"])
    
    # get up to three reviews for each hotel
    review_response = yelp_api.reviews_query(id=business["id"], limit=3)
    reviews = [review["text"] for review in review_response["reviews"]]
    
    # analyze the sentiment of each review
    review_sentiments = []
    for review in reviews:
        sentiment_scores = analyzer.polarity_scores(review)
        review_sentiments.append(sentiment_scores)
    
    # append the hotel data and review sentiments to the hotel_data list
    hotel_data.append([name, rating, review_count, address, reviews, review_sentiments])

# create a pandas dataframe with the hotel data
df = pd.DataFrame(hotel_data, columns=["Hotel Name", "Rating", "Review Count", "Address", "Reviews", "Review Sentiments"])

# create a stacked bar chart of sentiment analysis
positive_counts = []
negative_counts = []
neutral_counts = []

for row in df['Review Sentiments']:
    positive = 0
    negative = 0
    neutral = 0
    for review in row:
        if review['compound'] >= 0.05:
            positive += 1
        elif review['compound'] <= -0.05:
            negative += 1
        else:
            neutral += 1
    positive_counts.append(positive)
    negative_counts.append(negative)
    neutral_counts.append(neutral)

# create a new dataframe with the sentiment analysis data
sentiment_df = pd.DataFrame({'Positive': positive_counts, 'Negative': negative_counts, 'Neutral': neutral_counts})

# combine the original dataframe with the sentiment analysis dataframe
merged_df = pd.concat([df, sentiment_df], axis=1)

# create the stacked bar chart
merged_df.plot(kind='bar', x='Hotel Name', y=['Positive', 'Negative', 'Neutral'], stacked=True, figsize=(12, 8))
plt.title('Sentiment Analysis of Hotel Reviews')
plt.xlabel('Hotel Name')
plt.ylabel('Number of Reviews')
plt.show()

