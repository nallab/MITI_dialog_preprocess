"""Preprocess tools for a set of dialogue
- extraction
    - tokenization
    - lemmatization
- vectorization
    - bag of words
    - and more

Requirement:
    ginza==3.1.2, https://github.com/megagonlabs/ginza
    scikit-learn==0.22.1
"""

import spacy, ginza

def ginza_nlp():
    nlp = spacy.load('ja_ginza')
    return nlp

def tokenize_ginza(dataset, tokenizer):
    """Tokenize the docs with a while space ' ' inclucing lemmatize.

    Args:
        dataset (list): a list of 'dialogue set (e.g., train/test)' created by divide_dialog().
        tokenizer (ginza.Japanese): created by ginza_nlp() as default.
    Returns:
        docs_lemma (list): a list of dialogue sepalated with a white space.
    
    >>> X_train = ['<NA>今回はどのような目的で来られましたか？', '禁煙したいのですが、なかなか続かなくて。']
    >>> X_test = ['3rd text', '4th doc', '5th data']
    >>> nlp = ginza_nlp()
    >>> docs_token = tokenize_ginza([X_train, X_test], nlp)
    >>> print(type(docs_token[0][0]), len(docs_token[0][0]))
    <class 'str'> 38
    >>> print(docs_token[0][0])
    ＜ NA ＞ 今回 は どのような 目的 で 来る られる ます た か ？
    """
    docs_lemma = []
    for one_set in dataset:
        temp = []
        for sample in one_set:
            target = tokenizer(sample)
            doc_lemma = ''
            for sent in target.sents:
                sent_lemma = ''
                for token in sent:
                    sent_lemma += token.lemma_ + ' '
                doc_lemma += sent_lemma.rstrip() # remove last white space
            temp.append(doc_lemma)
        docs_lemma.append(temp)
    return docs_lemma

def bag_of_words(tokenized_dataset):
    """Make BoW vectors based on the 1st item in tokenized_dataset

    Args:
        tokenized_dataset (list): a set of tokenized dataset such as X_train, X_test.
            e.g., [['tokenized_doc1', 'tokenized_doc2'],['tokenized_doc3'],['tokenized_doc4']]
    Returns:
        vectors (list): a set of vector.
        vocabs (dict): vocabraries made by CountVectorizer.fit().
    >>> train_docs = ['1st doc', '2nd one']
    >>> test_docs = ['3rd text', '4th doc', '5th data']
    >>> vectors, vocabs = bag_of_words([train_docs, test_docs])
    >>> print(vocabs.vocabulary_)
    {'1st': 0, 'doc': 2, '2nd': 1, 'one': 3}
    >>> print(vectors[0][0].toarray())
    [[1 0 1 0]]
    """
    if len(tokenized_dataset[0]) <= 0:
        print("must consist of a set of tokeneized documents.")
        exit()
    target_doc = tokenized_dataset[0]

    from sklearn.feature_extraction.text import CountVectorizer
    stop = ginza.STOP_WORDS
    vectorizer = CountVectorizer(stop_words=stop)
    vocabs = vectorizer.fit(target_doc)
    vectors = []
    for dataset in tokenized_dataset:
        vec = vocabs.transform(dataset)
        vectors.append(vec)
    return vectors, vocabs

if __name__ == '__main__':
    import datasets
    all_df, documents_df = datasets.load_MITI_dialog("../dataset/example-20200312/", "csv", num_pre_context=1)
    X_docs, X_docs, y_train, y_test = datasets.divide_dialog(all_df, train_rate=0.8)

    nlp = ginza_nlp()
    tokenized_dataset = tokenize_ginza([X_docs, X_docs], nlp)
    print(type(tokenized_dataset[0][0]), len(tokenized_dataset[0][0]))
    print(tokenized_dataset[0][0])

    vectors, vectorizer = bag_of_words(tokenized_dataset)
    print(len(vectors), len(vectorizer.vocabulary_))
    print(vectorizer.vocabulary_)
    print(vectors[0][0])

    #X_train = vectors[0], X_test = vectors[0]


