import openai
import numpy as np  # standard math module for python
from pprint import pprint

# function to open the text file where the API key is stored


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# function to get the embedding vector of a given content using the GPT-3 engine


def gpt3_embedding(content, engine='text-similarity-babbage-001'):
    # removing any unwanted characters
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    # getting the embedding vector
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

# function to calculate the similarity of two vectors


def similarity(v1, v2):
    return np.dot(v1, v2)


# loading the API key
openai.api_key = open_file('openaiapikey.txt')

# function to match a given vector with a set of pre-defined categories


def match_class(vector, classes):
    results = list()
    for c in classes:
        score = similarity(vector, c['vector'])
        info = {'category': c['category'], 'score': score}
        results.append(info)
    return results


if __name__ == '__main__':
    # list of pre-defined categories
    categories = ['plant', 'reptile', 'mammal',
                  'fish', 'bird', 'pet', 'wild animal']
    classes = list()
    # getting the embedding vector for each category
    for c in categories:
        vector = gpt3_embedding(c)
        info = {'category': c, 'vector': vector}
        classes.append(info)
    # infinite loop to take input from user and match it with pre-defined categories
    while True:
        a = input('Enter a lifeform here: ')
        vector = gpt3_embedding(a)
        result = match_class(vector, classes)
        pprint(result)
