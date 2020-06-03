"""Preprocess tools for a set of dialogue
- extraction
    - tokenization
    - lemmatization
- vectorization
    - bag of words
    - and more

Requirement:
    ginza-3.1.2, https://github.com/megagonlabs/ginza
"""

import spacy

def ginza_nlp():
    nlp = spacy.load('ja_ginza')
    return nlp

def tokenize_ginza(docs, toknizer):
    """Tokenize the docs with a while space ' ' inclucing lemmatize.

    Args:
        docs (list): a list of dialogue created by divide_dialog().
        toknizer (ginza.Japanese): created by ginza_nlp().
    Returns:
        docs_lemma (list): a list of dialogue sepalated with a white space.
    
    >>> import datasets
    >>> all_df, documents_df = datasets.load_MITI_dialog('../dataset/example-20200312/', 'csv', 1)
    reading done for ../dataset/example-20200312/case5.csv
    reading done for ../dataset/example-20200312/case4.csv
    reading done for ../dataset/example-20200312/case1.csv
    reading done for ../dataset/example-20200312/case3.csv
    reading done for ../dataset/example-20200312/case2.csv
    >>> X_train, X_test, y_train, y_test = datasets.divide_dialog(all_df, train_rate=0.8, seed=0)
    >>> nlp = ginza_nlp()
    >>> docs_token = tokenize_ginza(X_train, nlp)
    >>> print(type(docs_token[0]), len(docs_token[0]))
    <class 'str'> 38
    >>> print(docs_token[0])
    ＜ NA ＞ 今回 は どのような 目的 で 来る られる ます た か ？
    """
    docs_lemma = []
    for doc in docs:
        target = toknizer(doc)
        doc_lemma = ''
        for sent in target.sents:
            sent_lemma = ''
            for token in sent:
                sent_lemma += token.lemma_ + ' '
            doc_lemma += sent_lemma.rstrip() # remove last white space
        docs_lemma.append(doc_lemma)
    return docs_lemma

if __name__ == '__main__':
    import datasets
    all_df, documents_df = datasets.load_MITI_dialog("../dataset/example-20200312/", "csv", num_pre_context=1)
    X_train, X_test, y_train, y_test = datasets.divide_dialog(all_df, train_rate=0.8)

    nlp = ginza_nlp()
    docs_token = tokenize_ginza(X_train, nlp)
    print(type(docs_token[0]), len(docs_token[0]))
    print(docs_token[0])

