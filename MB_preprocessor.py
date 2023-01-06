import pandas as pd
import re
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

mb_data = pd.read_csv("mbti_1.csv")

preprocessed_mb_data = pd.DataFrame(data = {"type":[], "posts":[]})

for index, row in mb_data.iterrows():
    print(index)
    row_type = row['type']
    row_posts = row['posts'][1:-1].split('|||')

    #Make each post sentence it's own entry
    row_type = [row_type for i in range(len(row_posts))]
    preprocessed_mb_data = preprocessed_mb_data.append(pd.DataFrame({'type':row_type, 'posts':row_posts}))

#Remove links from each entry
preprocessed_mb_data['posts'] = preprocessed_mb_data['posts'].apply(lambda x: re.sub(r'http.*?[\s+]', '', x + ' '))

#Remove numbers from each entry
preprocessed_mb_data['posts'] = preprocessed_mb_data['posts'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

#Get rid of all entries with less than 3 words
preprocessed_mb_data['word count'] = preprocessed_mb_data["posts"].apply(lambda x: len(re.findall(r'\w+', x)))
preprocessed_mb_data = preprocessed_mb_data[preprocessed_mb_data['word count'] >= 3]

#Make all entries lowercase
preprocessed_mb_data["posts"] = preprocessed_mb_data["posts"].apply(lambda x: x.lower())

# **If we graph the counts of the types, we can see that we have a larger proportion of introverts to extroverts**
# preprocessed_mb_data['type'].value_counts().plot(kind='bar')
# plt.show()

# **Apply sentiment analysis to the data as an additional feature**
mb_posts = preprocessed_mb_data.posts.values.tolist()

sentiment_classifier = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
sentiment_classification = sentiment_classifier(mb_posts, return_all_scores=True)


sentiment_data=[]
i=0
for x in sentiment_classification:
    print(i)
    sentiment_data.append([x[0]['score'], x[1]['score'], x[2]['score']])
    i+=1

preprocessed_mb_data['sentiment'] = sentiment_data

preprocessed_mb_data.to_csv('preprocessed_mbti.csv', sep='\t')