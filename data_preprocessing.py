import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
from scipy.sparse import coo_matrix

from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

tokenizer = WordPunctTokenizer()
stemmer = PorterStemmer()
lmtzr = WordNetLemmatizer()

cachedStopWords = stopwords.words("english")


# In[2]:


def replace_special_character(document):
    result = re.sub('[^a-zA-Z\n\.]', ' ', document).replace('.', ' ')
    result = ' '.join(result.split())
    result = "".join(result.splitlines())
    result=re.sub(r'\b\w{1,3}\b', '', result)
    return result.strip()


# In[3]:

def removestopword(document):
    text = ' '.join([word for word in document.strip().lower().split() if word not in cachedStopWords])
    return text


# In[4]:

def readTestFile():
    file = 'data.csv'
    data_frame = pd.read_csv(file, names=['label', 'text'])
    print('finished reading files ... ')
    data_frame['text'] = data_frame['text'].apply(lambda x : replace_special_character(x))
    print('finished cleaning...')
    return data_frame


# In[5]:

data = readTestFile()


# In[6]:

data.head()


# In[7]:

def pre_process(strng, enable_trivial=False):
    strng = re.sub('[^a-zA-Z ]', '', strng)
    words = []
    for i in strng.lower().split():
        i = lmtzr.lemmatize(i)
        i = str(i)
        words.append(i)
    strng = words
    return strng

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = pre_process(text)
    return tokens


# In[8]:

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', lowercase = True,tokenizer=tokenize_and_stem,
                     token_pattern = r'\b[a-zA-Z]+\b',ngram_range=(1,2),norm='l2')


# In[9]:

matrix =  tf.fit_transform(data['text'].tolist())
features = tf.get_feature_names() 


# In[11]:

matrix


# In[12]:

features[:10]


# In[ ]:


