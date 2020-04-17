# Marco Georgaklis, April 2020
# Vanderbilt British Periodicals Word Embedding Exercise

# we are importing Word2Vec, which allows us to create a neural network that assigns vectors to
# word meanings. We are also using Pandas DataFrame to access the text in the British Periodicals.
from gensim.models import Word2Vec
import pandas as pd
from pandas import DataFrame

# The struct we will be adding all of our text to
articles = []

# We are using pandas to read the csv and rename the columns
data = pd.read_csv("pbp.csv")
data.columns = ['article', 'journal', 'volume', 'issue', 'date', 'pages', 'url', 'text']

# we only need the text so we are making a new DataFrame with only the text column
text = DataFrame(data, columns=['text'])

# Not sure why the DataFrame only has 9 rows
print(len(text))

# It seems like the DataFrame only has 9 rows, so the loop is only running 9 times, but each
# iteration through the loop adds the article text to the articles struct and splits the article
# into individual words
for i in range(len(text)):
    articles += [text.ix[i]['text'].split()]

# We have to train the model so we feed in the articles we just searched into the Word2Vec
model = Word2Vec(articles, min_count=1)

# summarize vocabulary
words = list(model.wv.vocab)

# Now all the words are vectorized, this is the vector for 'man'
print(model['man'])

# save the model
model.save('model.bin')

# load the model
new_model = Word2Vec.load('model.bin')

# We can now use most_similar to find the word vectors closest to the different words we plug in
print(model.most_similar(positive=['men'], topn=5))
print(model.most_similar(positive=['women'], topn=5))
print(model.most_similar(positive=['king'], topn=5))

# We can also add and subtract words, in the example I used to learn from, they added king and
# woman and subtracted man to get queen. You can see from the output that we aren't quite there,
# but we likely will have associations that make more sense when we use more than 9 articles.
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)
