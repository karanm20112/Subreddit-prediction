"""
Name: Karan pankaj Makhija and Jeet Thakur
Version: Python 2.7
Title: Word Cloud
"""

from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import numpy as np
import requests

#Loading all the text from different csv's
first_csv = pd.read_csv('datascience.csv',encoding='UTF-8')
second_csv = pd.read_csv('cooking.csv',encoding='UTF-8')
third_csv = pd.read_csv('GRE.csv',encoding='UTF-8')


text =' '
text1 = ' '
text2 = ' '
#Removing the stop words
stopwords = set(STOPWORDS)
stopwords.update(['will','new','thisismissingtext'])

for x in first_csv.body:
    if type(x) is float:
        x = str(x).encode('utf-8')
    else:
        x = x.encode('utf-8')
    values = x.split()
    #Convert all the words to lower case
    for i in range(len(values)):
        values[i] = values[i].lower()

    for words in values:
        #Add it to a string
        text = text+words+' '
#Same goes for the second one
for x in second_csv.body:
    if type(x) is float:
        x = str(x).encode('utf-8')
    else:
        x = x.encode('utf-8')
    values = x.split()
    for i in range(len(values)):
        values[i] = values[i].lower()

    for words in values:
        text1 = text1+words+' '

#Creating text for third reddit
for x in third_csv.body:
    if type(x) is float:
        x = str(x).encode('utf-8')
    else:
        x = x.encode('utf-8')
    values = x.split()
    for i in range(len(values)):
        values[i] = values[i].lower()

    for words in values:
        text2 = text2+words+' '

#Putting the word cloud in a mask of our choice
mask1 = np.array(Image.open('brain.png'))
wc =WordCloud(max_words = 200,width = 1500,height= 1100, background_color = 'White',stopwords=stopwords,mask=mask1,contour_width
              =3, contour_color = 'blue', min_font_size = 10).generate(text)
#Plotting the figure
plt.figure(figsize = (40,40))
plt.imshow(wc)
plt.axis("off")
plt.savefig('img1.png')
#For second subreddit
mask2 = np.array(Image.open(requests.get('http://www.clker.com/cliparts/O/i/x/Y/q/P/yellow-house-hi.png', stream=True).raw))
wc =WordCloud(max_words = 200,width = 1500,height= 1100, background_color = 'White',stopwords=stopwords,mask=mask2,contour_width
              =3, contour_color = 'blue', min_font_size = 10).generate(text1)

plt.figure(figsize = (40,40))
plt.imshow(wc)
plt.axis("off")
plt.savefig('img2.png')

#For third subreddit
mask3 = np.array(Image.open('GRE.png'))
wc =WordCloud(max_words = 200,width = 1500,height= 1100, background_color = 'White',stopwords=stopwords,mask=mask3,contour_width
              =3, contour_color = 'blue', min_font_size = 10).generate(text2)
plt.figure(figsize = (40,40))
plt.imshow(wc)
plt.axis("off")
plt.savefig('img3.png')

