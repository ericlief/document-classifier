from nltk.corpus import reuters
import nltk as nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score,precision_score,recall_score
from nltk.corpus import stopwords, reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier


def collection_stats():
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents")

    train_docs = list(filter(lambda doc: doc.startswith("train"),
                        documents));
    print(str(len(train_docs)) + " total train documents")

    test_docs = list(filter(lambda doc: doc.startswith("test"),
                       documents))
    print(str(len(test_docs)) + " total test documents")

    # List of categories
    categories = reuters.categories()
    print(str(len(categories)) + " categories")

    # Documents in a category
    category_docs = reuters.fileids("acq")

    # Words for a document
    document_id = category_docs[0]
    document_words = reuters.words(category_docs[0])
    print(document_words)

    # Raw document
    print(reuters.raw(document_id))
    tokens = []
    for docid in train_docs:
        t = tokenize(reuters.raw(docid))
        tokens.extend(t)
    print(tokens[0])
    v = set(tokens)
    print("number of terms=", len(tokens))
    print("voc size=", len(v))


def tokenize(text):
    min_length = 3
    cachedStopWords = stopwords.words("english")
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token),
                                   words)))
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter(lambda token: p.match(token) and
                                   len(token) >= min_length,
                                   tokens))
    return filtered_tokens

# collection_stats()
# nltk.download()



# List of document ids
stop_words = stopwords.words("english")
documents = reuters.fileids()
train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                           documents))
train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

# Tokenisation
vectorizer = TfidfVectorizer(stop_words=stop_words,
                             tokenizer=tokenize)

# print(vectorizer)

# Learn and transform train documents
vectorised_train_documents = vectorizer.fit_transform(train_docs) # return term-doc matrix
#print("number of terms =", vectorised_train_documents.getnnz()
#print("number of doc =", vectorised_train_documents[0].
vectorised_test_documents = vectorizer.transform(test_docs)  # return doc-term matrix

print(vectorised_train_documents[:10])
print(vectorised_test_documents)


# Transform multilabel labels
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([reuters.categories(doc_id)
                                  for doc_id in train_docs_id])
test_labels = mlb.transform([reuters.categories(doc_id)
                             for doc_id in test_docs_id])

#print(train_labels[:10])
#print(reuters.categories(train_docs_id[0]))
#print(mlb.classes_)


mlp = MLPClassifier(solver='sgd', activation='logistic', hidden_layer_sizes=(10398,))
mlp.fit(vectorised_train_documents, train_labels)
predictions = mlp.predict(vectorised_test_documents)


"""
# Classifier
classifier = OneVsRestClassifier(LinearSVC(random_state=42))
classifier.fit(vectorised_train_documents, train_labels)

predictions = classifier.predict(vectorised_test_documents)
print(predictions)
"""

with open('results.txt', 'wt') as f:

    precision = precision_score(test_labels, predictions,
                                average='micro')
    recall = recall_score(test_labels, predictions,
                          average='micro')
    f1 = f1_score(test_labels, predictions, average='micro')

    print("Micro-average quality numbers")
    res = "Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}\n".format(precision, recall, f1)
    print(result)
    f.write("Micro-average quality numbers\n")
    f.write(res)

    precision = precision_score(test_labels, predictions,
                                average='macro')
    recall = recall_score(test_labels, predictions,
                          average='macro')
    f1 = f1_score(test_labels, predictions, average='macro')

    print("Macro-average quality numbers")
    res = "Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}\n".format(precision, recall, f1)
    print(result)
    f.write("Micro-average quality numbers\n")
    f.write(res)

    # Precision: 0.6493, Recall: 0.3948, F1-measure: 0.4665
