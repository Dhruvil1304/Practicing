import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict

plt.style.use(style='seaborn')

df = pd.read_csv('all-data.csv', encoding="ISO-8859-1")
print(df.head())
print(df.info())
print(df.isna().sum())
print(df['neutral'].value_counts())

y = df['neutral'].values
y.shape

x = df[
    'According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .'].values
x.shape

(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.4)
x_train.shape
y_train.shape
x_test.shape
y_test.shape

df1 = pd.DataFrame(x_train)
df1 = df1.rename(columns={0: 'news'})
df2 = pd.DataFrame(y_train)
df2 = df2.rename(columns={0: 'sentiment'})
df_train = pd.concat([df1, df2], axis=1)

print("############### Training #################")
print(df_train.head())
print(df_train.info)

df3 = pd.DataFrame(x_test)
df3 = df3.rename(columns={0: 'news'})
df4 = pd.DataFrame(y_test)
df4 = df2.rename(columns={0: 'sentiment'})
df_test = pd.concat([df3, df4], axis=1)

print('################ Test #################')
print(df_test.head())


# defining the function to remove punctuation
def remove_punctuation(text):
    if (type(text) == float):
        return text
    ans = ""
    for i in text:
        if i not in string.punctuation:
            ans += i
    return ans


# storing the puntuation free text in a new column called clean_msg
df_train['news'] = df_train['news'].apply(lambda x: remove_punctuation(x))
df_test['news'] = df_test['news'].apply(lambda x: remove_punctuation(x))

print(df_train.head())

print("################ Begin N-grams ###################")
'''
print('################ Example for N-grams #################')
 
example_sent = """This is a sample sentence,
                  showing off the stop words filtration."""

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)
# converts the words in word_tokens to lower case and then checks whether
# they are present in stop_words or not
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
# with no lower case conversion
filtered_sentence = []

for w in word_tokens:
  if w not in stop_words:
    filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)

def generate_N_grams(words,ngram):
  temp=zip(*[words[i:] for i in range(0,ngram)])
  ans=[' '.join(ngram) for ngram in temp]
  return ans

print(generate_N_grams(filtered_sentence,2))
'''


def generate_N_grams(text, ngram=1):
    words = [word for word in text.split(" ") if word not in set(stopwords.words('english'))]
    #print("Sentence after removing stopwords:", words)
    temp = zip(*[words[i:] for i in range(0, ngram)])
    ans = [' '.join(ngram) for ngram in temp]
    return ans


positiveValues = defaultdict(int)
negativeValues = defaultdict(int)
neutralValues = defaultdict(int)
# get the count of every word in both the columns of df_train and df_test dataframes

# get the count of every word in both the columns of df_train and df_test dataframes where sentiment="positive"
for text in df_train[df_train.sentiment == "positive"].news:
    for word in generate_N_grams(text):
        positiveValues[word] += 1
# get the count of every word in both the columns of df_train and df_test dataframes where sentiment="negative"
for text in df_train[df_train.sentiment == "negative"].news:
    for word in generate_N_grams(text):
        negativeValues[word] += 1
# get the count of every word in both the columns of df_train and df_test dataframes where sentiment="neutral"
for text in df_train[df_train.sentiment == "neutral"].news:
    for word in generate_N_grams(text):
        neutralValues[word] += 1
# focus on more frequently occuring words for every sentiment=>
# sort in DO wrt 2nd column in each of positiveValues,negativeValues and neutralValues
df_positive = pd.DataFrame(sorted(positiveValues.items(), key=lambda x: x[1], reverse=True))
print(df_positive)
df_negative = pd.DataFrame(sorted(negativeValues.items(), key=lambda x: x[1], reverse=True))
df_neutral = pd.DataFrame(sorted(neutralValues.items(), key=lambda x: x[1], reverse=True))
pd1 = df_positive[0][:10]
pd2 = df_positive[1][:10]
ned1 = df_negative[0][:10]
ned2 = df_negative[1][:10]
nud1 = df_neutral[0][:10]
nud2 = df_neutral[1][:10]

plt.figure(1, figsize=(16, 4))
plt.bar(pd1, pd2, color='green', width=0.4)
plt.xlabel("Words in positive dataframe")
plt.ylabel("Count")
plt.title("Top 10 words in positive dataframe-UNIGRAM ANALYSIS")
plt.savefig("positive-unigram.png")
plt.show()

plt.figure(1,figsize=(16,4))
plt.bar(ned1,ned2, color ='red', width = 0.4)
plt.xlabel("Words in negative dataframe")
plt.ylabel("Count")
plt.title("Top 10 words in negative dataframe-UNIGRAM ANALYSIS")
plt.savefig("negative-unigram.png")
plt.show()

plt.figure(1,figsize=(16,4))
plt.bar(nud1,nud2, color ='yellow', width = 0.4)
plt.xlabel("Words in neutral dataframe")
plt.ylabel("Count")
plt.title("Top 10 words in neutral dataframe-UNIGRAM ANALYSIS")
plt.savefig("neutral-unigram.png")
plt.show()

positiveValues2=defaultdict(int)
negativeValues2=defaultdict(int)
neutralValues2=defaultdict(int)
#get the count of every word in both the columns of df_train and df_test dataframes
#get the count of every word in both the columns of df_train and df_test dataframes where sentiment="positive"
for text in df_train[df_train.sentiment=="positive"].news:
  for word in generate_N_grams(text,2):
    positiveValues2[word]+=1
#get the count of every word in both the columns of df_train and df_test dataframes where sentiment="negative"
for text in df_train[df_train.sentiment=="negative"].news:
  for word in generate_N_grams(text,2):
    negativeValues2[word]+=1
#get the count of every word in both the columns of df_train and df_test dataframes where sentiment="neutral"
for text in df_train[df_train.sentiment=="neutral"].news:
  for word in generate_N_grams(text,2):
    neutralValues2[word]+=1
#focus on more frequently occuring words for every sentiment=>
#sort in DO wrt 2nd column in each of positiveValues,negativeValues and neutralValues
df_positive2=pd.DataFrame(sorted(positiveValues2.items(),key=lambda x:x[1],reverse=True))
df_negative2=pd.DataFrame(sorted(negativeValues2.items(),key=lambda x:x[1],reverse=True))
df_neutral2=pd.DataFrame(sorted(neutralValues2.items(),key=lambda x:x[1],reverse=True))
pd1bi=df_positive2[0][:10]
pd2bi=df_positive2[1][:10]
ned1bi=df_negative2[0][:10]
ned2bi=df_negative2[1][:10]
nud1bi=df_neutral2[0][:10]
nud2bi=df_neutral2[1][:10]

plt.figure(1,figsize=(16,4))
plt.bar(pd1bi,pd2bi, color ='green',width = 0.4)
plt.xlabel("Words in positive dataframe")
plt.ylabel("Count")
plt.title("Top 10 words in positive dataframe-BIGRAM ANALYSIS")
plt.savefig("positive-bigram.png")
plt.show()

plt.figure(1,figsize=(16,4))
plt.bar(ned1bi,ned2bi, color ='red', width = 0.4)
plt.xlabel("Words in negative dataframe")
plt.ylabel("Count")
plt.title("Top 10 words in negative dataframe-BIGRAM ANALYSIS")
plt.savefig("negative-bigram.png")
plt.show()

plt.figure(1,figsize=(16,4))
plt.bar(nud1bi,nud2bi, color ='yellow', width = 0.4)
plt.xlabel("Words in neutral dataframe")
plt.ylabel("Count")
plt.title("Top 10 words in neutral dataframe-BIGRAM ANALYSIS")
plt.savefig("neutral-bigram.png")
plt.show()
