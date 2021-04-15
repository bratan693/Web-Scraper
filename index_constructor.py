import collections
from bs4 import BeautifulSoup
import io
# README FILE
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import math
import json

from numpy import dot


stopWords = ["a", "about",  "above", "after", "again", "against", "all", "aman", "and", "anyare", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between",  "both", "but", "by", "can't",
             "cannot",  "could" "couldn't",  "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "hasn't", "have",
             "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
             "i'll",  "i'm", "i've", "if", "in", "into", "is", "isn't", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
             "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
             "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very",
             "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
             "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
             ]
#specialTags = ["title", "h1","h2","h3","b"] #
#include= ["body","li","strong","div","p"]
class invertedIndex:
    def __init__(self):
        self.words = collections.defaultdict(int)  # holds words and frequenc

    def bonusWeights(self, strings, bonusWeight, invIndex, docid,bonusWeightDictionary):
        lemmatizer = WordNetLemmatizer()
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        for tag in strings:
            words = tokenizer.tokenize(tag.text)
            for word in words:
                word = lemmatizer.lemmatize(word.lower())
                if word.isalpha() == True:
                    if word not in stopWords and len(word) != 1:
                        if word not in invIndex:
                            invIndex[word] = {}
                            invIndex[word][docid] = 1
                            bonusWeightDictionary[word] = {}
                            bonusWeightDictionary[word][docid] = bonusWeight
                        elif word in invIndex:
                            if docid not in invIndex[word]: 
                                invIndex[word][docid] = 1
                                bonusWeightDictionary[word][docid] = bonusWeight
                            else:
                                invIndex[word][docid] += 1
                                bonusWeightDictionary[word][docid] = bonusWeight

    def regularWeights(self, strings, RegularWeight, invIndex, docid,bonusWeightDictionary):
        lemmatizer = WordNetLemmatizer()
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        for tag in strings:
            words = tokenizer.tokenize(tag.text)
            for word in words:
                word = lemmatizer.lemmatize(word.lower())
                if word.isalpha() == True:
                    if word not in stopWords and len(word) != 1:
                        if word not in invIndex:
                            invIndex[word] = {}
                            invIndex[word][docid] = RegularWeight
                            bonusWeightDictionary[word] = {}
                            bonusWeightDictionary[word][docid] = 0
                        elif word in invIndex:
                            if docid not in invIndex[word]:
                                invIndex[word][docid] = RegularWeight
                                bonusWeightDictionary[word][docid] = 0
                            else:
                                invIndex[word][docid] += 1
                                bonusWeightDictionary[word][docid] = 0 

    def parseWords(self, file, docid, invIndexDictionary, url, bonusWeightDictionary):
        with io.open(file, 'r', encoding = 'utf-8') as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            titles = soup.find_all('title')
            self.bonusWeights(titles, 3.5, invIndexDictionary,docid,bonusWeightDictionary)
            strongTag = soup.find_all('strong')
            self.bonusWeights(strongTag,3,invIndexDictionary,docid,bonusWeightDictionary)
            # words under title are more important than h1,h2,h3, b tags
            headers = soup.find_all('h1')
            headers += soup.find_all('h2')
            headers += soup.find_all('h3')
            headers += soup.find_all('b')
            self.bonusWeights(headers, 2.5, invIndexDictionary, docid,bonusWeightDictionary)
            # parse everything else thats not in title, h1, h2, h3, b here
            normalTags = soup.find_all('body')
            normalTags += soup.find_all('li')    
            normalTags += soup.find_all('div') 
            normalTags += soup.find_all('p')       
            self.regularWeights(normalTags, 1, invIndexDictionary,docid,bonusWeightDictionary)

    # translate the DOCID to URL for quick look-up on query
    def getUrl(self):
        file= "WEBPAGES_RAW/bookkeeping.json"
        urlLinks = collections.defaultdict(str)
        with open(file) as f:
            data = json.load(f)
        for key, value in data.items():
            if key not in urlLinks:
                urlLinks[key] = value
        pickle_in = open("docToUrl.pickle", "wb")
        pickle.dump(urlLinks,pickle_in)
        pickle_in.close()


    ## Creates pickle file for dictionary of words: {doc,frequency}
    def pickleAllWords(self,dict1):
        pickle_out = open("mainDictionary.pickle", "wb") 
        pickle.dump(dict1,pickle_out)
        pickle_out.close()

    def pickleBonusWeights(self,dict1):
        pickle_out = open("BonusWeights.pickle", "wb") 
        pickle.dump(dict1,pickle_out)
        pickle_out.close()
        
    def calculateTF(self):
        tfScore = collections.defaultdict(dict)
        pickle_in = open("mainDictionary.pickle","rb")
        indexContainer = pickle.load(pickle_in)
        pickle_in.close()
        for key,value in indexContainer.items():
            for i in value:
                # ={}
                val = indexContainer[key][i]
                tfScore[key][i] = 1+math.log10(val) 
        createTF = open("tfScore.pickle", "wb")
        pickle.dump(tfScore,createTF)
        createTF.close()



    def calculateIDF(self):
        pickle_in = open("tfScore.pickle","rb")
        dictionary = pickle.load(pickle_in)
        pickle_in.close()
        idfDict = collections.defaultdict(int)
        for key,value in dictionary.items():
            count = 0
            length = len(dictionary[key])
            idf = math.log10(37497 / length)
            idfDict[key] = idf
        pickle_in = open("idfScore.pickle", "wb")
        pickle.dump(idfDict, pickle_in)
        pickle_in.close()
  
  
    def createWeight(self):
        tfPickle = open("tfScore.pickle", "rb")
        tfDict = pickle.load(tfPickle)
        tfPickle.close()
        
        idfPickle = open("idfScore.pickle", "rb")
        idfDict = pickle.load(idfPickle)
        idfPickle.close()
        weight = collections.defaultdict(dict)

        for key,value in tfDict.items():
            for docID in value:
                weight[key][docID] = tfDict[key][docID] * idfDict[key]
        for key,value in weight.items():
            weight[key] = dict(sorted(value.items(), key = lambda item: item[1], reverse= True))
            
        scorePickle = open("weights.pickle", "wb")
        pickle.dump(weight,scorePickle)
        scorePickle.close()

    
    def calculateBonusWeights(self):
        pickle_object = open("BonusWeights.pickle", "rb")
        bonusWeights = pickle.load(pickle_object)
        pickle_object.close()

        pickle_object = open("weights.pickle","rb")
        tfIdfWeights = pickle.load(pickle_object)
        pickle_object.close()

        newWeights = collections.defaultdict(dict)

        for key, value in tfIdfWeights.items():
            for docID in value:
                    newWeights[key][docID] = bonusWeights[key][docID] + tfIdfWeights[key][docID]
        for key,value in newWeights.items():
            newWeights[key] = dict(sorted(value.items(), key = lambda item: item[1], reverse= True))
      
        pickle_object = open("ModifiedWeights.pickle", "wb")
        pickle.dump(newWeights,pickle_object)
        pickle_object.close()


    def computeCosine():
        print("LOL")

    def getQuery(self, query):
        lemmatizer = WordNetLemmatizer()
        # open pickles
        # get a map of document ids to the url
        pickle_object = open("docToUrl.pickle", "rb")
        docToUrl = pickle.load(pickle_object)
        pickle_object.close()
        # Bonus weights for special tags
        pickle_object = open("BonusWeights.pickle", "rb")
        bonusweights = pickle.load(pickle_object)
        pickle_object.close()
        # Modified weights 
        pickle_object = open("ModifiedWeights.pickle", "rb")
        tfIdfWeights = pickle.load(pickle_object)
        pickle_object.close()

        while True:
            if query == "quit":
                quit()
            else:
                queryIndexer = collections.defaultdict(int)
                numOfDocuments = 0
                Lowerquery = lemmatizer.lemmatize(query.lower())
                splitWords= Lowerquery.split(" ")
                queries = query.split(" ")
                for s in splitWords: 
                    if s in stopWords:
                        splitWords.remove(s)
                lengthofQueries = len(splitWords)
                # QUERY IF NUM WORDS IS GREATER THAN 1
                cosine = 0
                if(len(splitWords) > 1):
                    normalizedScores = collections.defaultdict(dict)
                    rawFreq = collections.defaultdict(int)
                    length = collections.defaultdict(dict)
                    # intializing a dictionary for each word in our query
                    for term in splitWords:
                        rawFreq[term] += 1
                    # finding the weights for each query word
                    for key in rawFreq: 
                        rawFreq[key] =  1 + math.log10(rawFreq[key]) 
                    idf = 0 
                    for term in rawFreq: 
                        if term in tfIdfWeights:
                            docLength = len(tfIdfWeights[term])
                            if docLength  >=1 :
                                idf = math.log10(37497/docLength)
                                rawFreq[term] *= idf
                            else:
                                rawFreq[term] = 0
                    queryLength = 0
                    # calculate the length of query, square each term and then sqrt the total at the end
                    for term in rawFreq:
                        queryLength += rawFreq[term] ** 2
                    queryLength = math.sqrt(queryLength) 
                    for term in rawFreq:
                        if queryLength == 0:
                            rawFreq[term] = 0
                        else: 
                            rawFreq[term] = rawFreq[term] / queryLength
                    amountofDocs = []
                    for word in splitWords:
                        if word not in amountofDocs:
                            amountofDocs.append(word)
                    for word in amountofDocs:
                        numOfDocuments += len(tfIdfWeights[word])
                    #NORMALIZE INDEX
                    #Gets all dictionary docid mapping to frequency to a new dictionary for normalization for relevant terms
                    for term in rawFreq: 
                        if term in tfIdfWeights:
                            normalizedScores[term] = tfIdfWeights.get(term)
                    for key,value in normalizedScores.items():
                        for docID in value:
                            length[key][docID] = normalizedScores[key][docID] ** 2
                    listofWt = collections.defaultdict(int)
                    tmplist = collections.defaultdict(int)
                    for value in length.values(): 
                        for docID, wt in value.items():
                            if docID not in tmplist:
                                tmplist[docID] =0
                            for term in rawFreq:
                                if tmplist[docID] != lengthofQueries:
                                    listofWt[docID] += wt
                                    tmplist[docID] +=1
                                else:
                                    break 
                    for key,value in normalizedScores.items(): 
                        for docID in value:
                            normalizedScores[key][docID] = normalizedScores[key][docID] / listofWt[docID]

                    # add bonsu weights from important tags to the final scores
                    for key, value in normalizedScores.items(): 
                        for docID in value:
                            normalizedScores[key][docID] += bonusweights[key][docID]
                

                    newtmp = []
                    # Get docIDs for each query so we can weigh and dot product it
                    for key,value in rawFreq.items():
                        newtmp.append(value) 
                    display = collections.defaultdict(int)
                    # Calculate the cosine score using all the values we got before
                    for value in normalizedScores.values():
                        for docID, wt in value.items():
                            b = []
                            if docID not in display:
                                 display[docID] =0
                            for term in rawFreq:
                                if len(b) != lengthofQueries:
                                    b.append(wt) 
                                else:
                                    break
                            display[docID] = dot(b,newtmp)
                            
                    displaySort = dict(sorted(display.items(), key=lambda item: item[1], reverse = True))
                    
                    count = 0
                    for docID in displaySort:
                        print("URL {}: {}".format(count+1, docToUrl[docID]))
                        count += 1
                        if count == 20:
                            break
                    print("Number of Documents:", numOfDocuments)
                    
                else:
                    count = 0
                    numOfDocuments = len(tfIdfWeights[Lowerquery])
                    for key,value in tfIdfWeights[Lowerquery].items():
                        print("URL {}: {}".format(count+1, docToUrl[key]))
                        count += 1
                        if count == 20:
                            break
                    if count == 0:
                        print("No results for", query)
                    print("Number of Documents:", numOfDocuments)
                
                query = input("Query: ")
