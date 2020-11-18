import numpy as np
from scipy import spatial

from datetime import datetime

max_doc_len = 500

time_start = datetime.now()
print("Starting time is " + str(time_start))
glove_embeddings = {}
embeds_file = open('glove/simple.txt', 'r')
#embeds_file = open('glove/glove.840B.300d.txt', 'r')
for line in embeds_file:
    try:
        splitLines = line.split()
        word = splitLines[0]
        vector = np.array([float(value) for value in splitLines[1:]])
        glove_embeddings[word] = vector
    except:
        continue
#print(len(glove_embeddings))
time_glove = datetime.now()
print("Glove completed " + str(time_glove) + " required " + str(time_glove - time_start))

def get_embedding(text):
    words = text.split(' ')
    count_words = 0
    text_vector = np.zeros((300), dtype='float32')
    mod_text = ""
    for word in words:
        if word in glove_embeddings:
            text_vector += glove_embeddings[word]
            count_words += 1
            mod_text += (word + ' ')
    if count_words > 0:
        text_vector /= count_words
    return [text_vector, mod_text]

queries = []
#queries_file = open('collection_queries/queries.dev.small.tsv', 'r')
queries_file = open('collection_queries/small_queries_test.tsv', 'r')
q_line = queries_file.readlines()[:10]
for q in q_line:
    queries.append((q.split('\t')[1]).rstrip())
#print(queries)

docs = []
docs_file = open('corpus/small.tsv', 'r')
#docs_file = open('corpus/msmarco-docs.tsv', 'r')
docs_lines = docs_file.readlines()[:100]
for doc in docs_lines:
    docs.append((doc.split('\t')[3]).rstrip())
#print(doc)

document_embeddings = []
document_lengths = []
modified_documents = []
for (i, doc) in enumerate(docs):
    sentences = doc.split('.')
    curr_doc = []
    curr_lengths = []
    mod_doc = []
    for sentence in sentences:
        embed, length = get_embedding(sentence)
        curr_doc.append(embed)
        curr_lengths.append(len(length))
        mod_doc.append(length)
    #print(curr_lengths)
    document_embeddings.append(curr_doc)        
    document_lengths.append(curr_lengths)
    modified_documents.append(mod_doc)
#print(document_lengths)
time_docs = datetime.now()
time_prev = datetime.now()
print("Document embeddings completed " + str(time_docs) + " required " + str(time_docs - time_glove))

#queries_vector = []
for (i, query) in enumerate(queries):
    query_vector, mod_query = get_embedding(query)
    #print(query_vector.shape)
    ids = i
    fw = open('docs/' + str(ids) + '.txt', 'w')
     
    for (j, doc) in enumerate(docs):
        sentences = doc.split('.')
        sentences_scores = [[0]*3 for _ in range(len(sentences))] # creating one for index, one for length and one for relevance score
        for (k, sentence) in enumerate(sentences):
            #sen_embed = get_embedding(sentence)
            #print(sen_embed.shape)
            sentences_scores[k][0] = k
            #sentences_scores[k][1] = len(sentence.split(' '))
            sentences_scores[k][1] = document_lengths[j][k]
            sentences_scores[k][2] = 1 - spatial.distance.cosine(query_vector, document_embeddings[j][k])
        sentences_scores = sorted(sentences_scores, key=lambda x: -x[2])
        #print("query num " + str(i) + " having sentence scores as  " + str(sentences_scores))
        #print("\n\n\n\n\n")
        final_doc = ""
        final_doc += mod_query
        new_doc_len = len(mod_query.split(' '))
        idx = 0
        while idx < len(sentences) and (new_doc_len + sentences_scores[idx][1]) < 512:
            final_doc += sentences[sentences_scores[idx][0]]
            new_doc_len += sentences_scores[idx][1]
            idx += 1
        #print(final_doc)
        fw.write(final_doc + '\n')
    time_query = datetime.now()
    print("query " + str(i) + " completed by " + str(time_query) + " took " + str(time_query - time_prev))
    time_prev = time_query
        
time_final = datetime.now()
print("final timing " + str(time_final))
#print(queries_vector)
