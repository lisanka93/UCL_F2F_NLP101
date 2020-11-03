# simple chatbot to demonstrate text similarity
# run with python -W ignore bot_2.py (python3 if not default)


import numpy as np
from nltk.stem import WordNetLemmatizer
import re

stop_words_file = 'SmartStoplist.txt'
#download them at https://nlp.stanford.edu/projects/glove/
glove_file = '/home/lisanka/Third-study/glove/glove.6B.100d.txt'


#a faster way to do it is to store it in a pickle file
def loadGloveModel(gloveFile):

    """ reads glove word vectors into dictionary"""

    print ("Loading Glove Model")
    with open(gloveFile, encoding="utf8" ) as f:
        content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    #print(model)
    return model

def loadStopwords(stopwordfile):

    """ reads stopwords into list """

    stop_words = []

    with open(stopwordfile, "r") as f:
        for line in f:
            stop_words.extend(line.split())

    return stop_words


def preprocess(raw_text):

    """ preprocesses given input by only keeping words, converting everything
    to lower case, splitting into tokens, lemmatising and deleting stopwords"""

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split
    words = letters_only_text.lower().split()

    # lemmatise and remove stopwords
    cleaned_words = []

    for word in words:
        lemmatizer = WordNetLemmatizer()
        word = lemmatizer.lemmatize(word)
        if word not in stop_words:
            cleaned_words.append(word)

    return cleaned_words


def cos_sim(a,b):

    """ manual implementation of cosine similarity calculation """

    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)

    return(cos)


def cosine_distance_wordembedding_method(s1, s2):

    """ vectorising input using glove model and calculating cosine similarity """

    try:
        #not a good implementation - gives error if word not in model
        #vector_1 = np.mean([model_glove[word] for word in preprocess(s1)],axis=0)
        #vector_2 = np.mean([model_glove[word] for word in preprocess(s2)],axis=0)

        #better implementation
        sent_vector_1 = []
        for word in preprocess(s1):
            if word not in model_glove:
                word_vector = np.array(np.random.uniform(-1.0, 1.0, 300))
                model_glove[word] = word_vector
                sent_vector_1.append(word_vector)
            else:
                word_vector = model_glove[word]
                sent_vector_1.append(word_vector)
        sent_vector_1 = np.array(sent_vector_1)
        vector_1 = np.mean(sent_vector_1, axis=0)

        sent_vector_2 = []
        for word in preprocess(s2):
            if word not in model_glove:
                word_vector = np.array(np.random.uniform(-1.0, 1.0, 300))
                model_glove[word] = word_vector
                sent_vector_2.append(word_vector)
            else:
                word_vector = model_glove[word]
                sent_vector_2.append(word_vector)
        sent_vector_2 = np.array(sent_vector_2)
        vector_2 = np.mean(sent_vector_2, axis=0)



        cosine = cos_sim(vector_1, vector_2)

        return round((cosine)*100,2)
    except :
        return 0

def Sort(sub_li):

    """ sorts a list of tuples by second element"""

    sub_li.sort(key = lambda x: x[1], reverse = True)
    return sub_li


samples = [
['Hi',
'Hello'],
['I need help',
'What do you need help with?'],
['I have a complaint.',
'Please elaborate your concern'],
['I need help setting up my workspace',
'What OS are you using?'],
['My wifi does not work',
'Have you checked whether your router is pluggedin?'],
['My laptop hangs',
'Have you tried restarting it you schmuck?'],
['Okay Thanks',
'No Problem! Have a Good Day!'],
['I struggle installing the printer',
'Have you downloaded all the necessary drivers?'],
['I keep on getting popups in my browser',
'Have you wateched porn again? *eyeroll*'],
['You are mean',
'Yes I know.'],
['I dont know how to set up outlook to receive e-mails from googlemail',
'Let me google that for you'],
['I cant connect to eduroam',
'Maybe you\'re not too smart for uni then']
]

model_glove = loadGloveModel(glove_file)
stop_words = loadStopwords(stop_words_file)

#testing that function works
s1 = "this sentence is about apples and oranges"
s2 = "i like cats and dog"

print(cosine_distance_wordembedding_method(s1,s2))


name=input("Enter Your Name: ")
print("Welcome to the Bot Service! Let me know how can I help you?")
while True:
    request=input(name+': ')
    if request=='Bye' or request =='bye':
        print('Bot: Bye')
        break
    else:
        possible_reponses = []
        # looping through dialogue samples and calculating similarity between
        # user input and any of the samples in our database
        for sam in samples:
            user_i = sam[0]
            bot_i = sam[1]
            cos = cosine_distance_wordembedding_method(request,user_i)

            # if sample with high similarity found, add to possible responses
            if cos > 70:
                possible_reponses.append([bot_i, cos])

        # if no similar samples found
        if len(possible_reponses) == 0:
            response = "Sorry, I don't understand"
        else:
        # if found, sort possible responses and output the one with the highest
        # similarity measure
            possible_reponses = Sort(possible_reponses)
            response= possible_reponses[0][0]
        print('Bot:',response)
