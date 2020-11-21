import numpy as np
import pandas as pd
import os
import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())
corpus = pd.read_csv(os.path.join('corpus', f'msmarco-docs.tsv'), sep='\t', header=None, names=['did', 'url', 'title', 'body'])

top100 = pd.read_csv('msmarco-doctrain-top100', sep=' ', header=None, names=['qid', 'q0', 'did', 'rank', 'score', 'model'])

glove_embeddings = {}
#embeds_file = open('glove/simple.txt', 'r')
embeds_file = open('glove/glove.840B.300d.txt', 'r')
for line in embeds_file:
    try:
        splitLines = line.split()
        word = splitLines[0]
        vector = np.array([float(value) for value in splitLines[1:]])
        glove_embeddings[word] = vector
    except:
        continue


queries_file = open(os.path.join('collection_queries', 'queries.train.tsv'))
queries = {}
queries_content = queries_file.readlines()
for line in queries_content:
    qid, query = line.split('\t')
    queries[qid] = query

def generate_Embedding(text, docid):
    sentences = text.split('.')
    #fembed = open('docs_generated/embeddings/' + docid + '.txt', 'w')
    ftext = open('docs_generated/text/' + docid + '.txt', 'w')
    embedding_array = np.zeros((len(sentences), 300), dtype='float32')
    for i, sentence in enumerate(sentences):
        words = sentence.split(' ')
        count_words = 0
        text_vector = np.zeros((300), dtype='float32')
        mod_text = ""
        for word in words:
            if word in glove_embeddings:
                embedding_array[i] += glove_embeddings[word]
                count_words += 1
                mod_text += (word + ' ')
        if count_words > 0:
            #text_vector /= count_words
            embedding_array[i] /= count_words
        #embedding_array
        #for i in range(text_vector.shape[0]):
        #    fembed.write(str(text_vector[i]) + '\t')
        #fembed.write('\n')
        ftext.write(mod_text + '\n')
    np.save(os.path.join('docs_generated', 'embeddings', docid + '.txt'), embedding_array)   
    #fembed.close()
    ftext.close()

f = open('msmarco-doctrain-selected-queries.txt', 'r')
selected_queries = f.readlines()[:10000]
for qid in selected_queries:
    qid = qid.rstrip()
    #print(qid)
    #docs_list = top100['qid'].values == qid
    docs_list = top100.loc[top100['qid'] == int(qid)]
    #docs_list = top100.query('qid == qid')
    #print(docs_list)
    for index, row in docs_list.iterrows():
        doc_id = row['did']
        #print(doc_id)
        if os.path.isfile(os.path.join('docs_generated', 'text', doc_id + '_embed.txt')) == False:
            data = corpus.loc[corpus['did'] == doc_id]
            #print(type(data['body'].values[0]))
            generate_Embedding(data['body'].values[0], doc_id)
    print(qid) 
f.close()
