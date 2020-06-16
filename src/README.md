# How to use
```Python
import datasets, preprocess

all_df, documents_df = datasets.load_MITI_dialog("../dataset/example-20200312/", "csv", num_pre_context=1)
X_train, X_test, y_train, y_test = datasets.divide_dialog(all_df, train_rate=0.8)

nlp = preprocess.ginza_nlp()
docs_token = preprocess.tokenize_ginza(X_train, nlp)
print(type(docs_token[0]), len(docs_token[0])) #=> <class 'str'> 38
print(docs_token[0]) #=> ＜ NA ＞ 今回 は どのような 目的 で 来る られる ます た か ？
```

<hr>

## 参考
- [SpaCy Tutorial](https://github.com/yuibi/spacy_tutorial/blob/master/README.md)
