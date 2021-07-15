import tensorflow
import tflearn
import pickle
import numpy
import nltk
from nltk.stem.lancaster import LancasterStemmer
import random
import json
import pandas as pd
from csv import DictWriter
from tensorflow.keras.layers import Dense, Activation, Dropout
stemmer = LancasterStemmer()
from tensorflow.keras.models import Sequential

row_list = []

'''Build Deep Learning Model'''
# Read CSV file
data = pd.read_csv('ChatBot-short.csv')

# Define list
data_words=[]
index_class=[]
x_data=[]
y_data=[]


temp_ques=[]

# Convert CSV data to JSON format so it can be processed and trained in Deep Learning Model
for i in data.Question:
    temp=[]
    temp.append(i)
    temp_ques.append(temp)

temp_ans=[]
for i in data.Answer:
    temp=[]
    temp.append(i)
    temp_ans.append(temp)

data['Question'] = temp_ques
data['Answer'] = temp_ans

result = data.to_json(orient="table")
parsed = json.loads(result)
print(parsed['data'])

# Processing Of Model
for row in parsed['data']:
    for question in row['Question']:
        question_data_words = nltk.word_tokenize(question)
        data_words.extend(question_data_words)
        x_data.append(question_data_words)
        y_data.append(row["index"])
    if row["index"] not in index_class:
        index_class.append(row["index"])

    data_words = [stemmer.stem(w.lower()) for w in data_words if w != "?"]
    data_words = sorted(list(set(data_words)))

    index_class = sorted(index_class)

    X = []
    Y = []
    out_empty = [0 for _ in range(len(index_class))]
    for x, doc in enumerate(x_data):
        bow = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in data_words:
            if w in wrds:
                bow.append(1)
            else:
                bow.append(0)

        out_data = out_empty[:]
        out_data[index_class.index(y_data[x])] = 1

        X.append(bow)
        Y.append(out_data)

    train_data = numpy.array(X)
    test_data = numpy.array(Y)
    with open("data.pickle", "wb") as f:
        pickle.dump((data_words, index_class, train_data, test_data), f)

# Developing model and save model
# tensorflow.reset_default_graph()
net_model = tflearn.input_data(shape=[None, len(train_data[0])])
net_model = tflearn.fully_connected(net_model, 8)
net_model = tflearn.fully_connected(net_model, 8)
net_model = tflearn.fully_connected(net_model, len(test_data[0]), activation="softmax")
net_model = tflearn.regression(net_model)

DNN_model = tflearn.DNN(net_model)

DNN_model.fit(train_data, test_data, n_epoch=500, batch_size=8, show_metric=True)
DNN_model.save("model.tflearn")

'''Generating Bag of Words for tokenizing and for numeric feature'''
def bag_data_words(s, data_words):
    bow = [0 for _ in range(len(data_words))]

    t_data_words = nltk.word_tokenize(s)
    t_data_words = [stemmer.stem(word.lower()) for word in t_data_words]

    for se in t_data_words:
        for i, w in enumerate(data_words):
            if w == se:
                bow[i] = 1
    return numpy.array(bow)

'''Chat section'''
def chat(inp):
    # print("Start talking with the bot (type quit to stop)!")
    user_input = inp
    count = 0

    # Adding Question and Answer if the message contains Question or Answer in it
    for i in user_input.split(" "):
        if(count == 1):
            ques = ques + " " + i
        if(i == "Question-"):
            ques = ""
            count=1

    if(count == 1):
        row_list.insert(0, ques)
        return "Please enter Answer."

    ans_count = 0
    for i in user_input.split(" "):
        if(ans_count == 1):
            ans = ans + " " + i
        if(i == "Answer-"):
            ans = ""
            ans_count=1

    if (ans_count == 1):
        # Storing data into dictionary
        dict_row = dict()
        dict_row['Question'] = row_list[0]
        dict_row['Answer'] = ans
        print(dict_row)

        # list of column names
        field_names = ['Question', 'Answer']

        # Appending into CSV data
        with open('ChatBot-short.csv', 'a') as fd:
            fd_object = DictWriter(fd, fieldnames=field_names)
            fd_object.writerow(dict_row)
            fd.close()

        df = pd.read_csv('ChatBot-short.csv')
        return "Question & Answer Added."
    # if user_input.lower() == "quit":
    #     break

    # Predicting model based on question asked
    out = DNN_model.predict([bag_data_words(user_input, data_words)])
    out_array = numpy.argmax(out)
    row_no = index_class[out_array]

    # Finding Answer based on index
    for d in parsed["data"]:
        if d['index'] == row_no:
            ans = d['Answer']

    return random.choice(ans)

