"""
Name : Karan Makhija and Jeet Thakur
Version: Python 2.7
Title: Extraction and creation of data
"""
import praw
import pandas as pd
import datetime as dt

def get_date(created):
    """
    Change the pattern of the date
    :param created: date
    :return: timestamp in a new way
    """
    return dt.datetime.fromtimestamp(created)

#Using the prawer to get the data.
reddit = praw.Reddit(client_id='T47warDvf1Wuow', client_secret="5_59h8RDb6VPa4DDrVvmR4IPJGI",
                     password='karanm2010', user_agent='FIS',
                     username='karanm2010')

#Taking three subreddits
subreddit = reddit.subreddit('datascience')
subreddit1 = reddit.subreddit('cooking')
subreddit2 = reddit.subreddit('GRE')

#Taking top 1000 posts
top_subreddit = subreddit.top(limit= 1000)
top_subreddit1 = subreddit1.top(limit = 1000)
top_subreddit2 = subreddit2.top(limit = 1000)

#Making dictionaries and then converting it to dataframes
topics_dict = { "title":[], \
                "score":[], \
                "id":[], "url":[], \
                "comms_num": [], \
                "created": [], \
                "body":[]}

topics_dict1 = { "title":[], \
                "score":[], \
                "id":[], "url":[], \
                "comms_num": [], \
                "created": [], \
                "body":[]}

topics_dict2 = { "title":[], \
                "score":[], \
                "id":[], "url":[], \
                "comms_num": [], \
                "created": [], \
                "body":[]}

#appending all the data in the dictionary which has body length greater than 1
for submission in top_subreddit:
    if len(submission.selftext)>=1:
        topics_dict["title"].append(submission.title)
        topics_dict["score"].append(submission.score)
        topics_dict["id"].append(submission.id)
        topics_dict["url"].append(submission.url)
        topics_dict["comms_num"].append(submission.num_comments)
        topics_dict["created"].append(submission.created)
        topics_dict["body"].append(submission.selftext)

for submission in top_subreddit1:
    if len(submission.selftext)>=1:
        topics_dict1["title"].append(submission.title)
        topics_dict1["score"].append(submission.score)
        topics_dict1["id"].append(submission.id)
        topics_dict1["url"].append(submission.url)
        topics_dict1["comms_num"].append(submission.num_comments)
        topics_dict1["created"].append(submission.created)
        topics_dict1["body"].append(submission.selftext)

for submission in top_subreddit2:
    if len(submission.selftext)>=1:
        topics_dict2["title"].append(submission.title)
        topics_dict2["score"].append(submission.score)
        topics_dict2["id"].append(submission.id)
        topics_dict2["url"].append(submission.url)
        topics_dict2["comms_num"].append(submission.num_comments)
        topics_dict2["created"].append(submission.created)
        topics_dict2["body"].append(submission.selftext)

#Creating pandas dataframe
topics_data = pd.DataFrame(topics_dict)
topics_data1 = pd.DataFrame(topics_dict1)
topics_data2 = pd.DataFrame(topics_dict2)

#Applying time stamp
_timestamp = topics_data["created"].apply(get_date)
_timestamp1 = topics_data1["created"].apply(get_date)
_timestamp2 = topics_data2["created"].apply(get_date)

#Assigning the target values to each dataframe
topics_data1 = topics_data1.assign(timestamp = _timestamp)
topics_data1['Target'] = 1
topics_data = topics_data.assign(timestamp = _timestamp)
topics_data["Target"] = 0
topics_data2 = topics_data2.assign(timestamp = _timestamp2)
topics_data2["Target"] = 2

#Making all the dataframe equals
topics_data_subset = topics_data.iloc[0:350]
topics_data1_subset = topics_data1.iloc[0:350]
topics_data2_subset = topics_data2.iloc[0:350]
merged_df = pd.concat([topics_data_subset,topics_data1_subset,topics_data2_subset])
topics_data.to_csv('datascience.csv',encoding='UTF-8')
topics_data1.to_csv('cooking.csv',encoding='UTF-8')
topics_data2.to_csv('GRE.csv',encoding='UTF-8')
merged_df.to_json('Final1.json',orient='records')

