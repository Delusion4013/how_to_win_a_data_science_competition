# Week 1 - Feature Extraction from text and images

## Conclusion

1. Texts
   1. Preprocessing
      1. Lowercase, stemming, lemmarization, stop words
   2. Bag of words
      1. Huge vectors
      2. Ngrams can help to use local context
      3. TFiDF can be of use as postprocessing
   3. Word2vec
      1. Relatively small vectors
      2. Pretrained models
2. Images
   1. Features can be extracted from different layers
   2. Careful choosing of pretrained network can help
   3. Finetuning allows to refine pretrained models
   4. Data augmentation can improve the model

## Text -> vector

### Pipeline of Bag of words

#### Text Preprocessing

##### Lowercase

- Reduce the number of possible columns

##### Lemmatization and stemming

- Stemming:

  democracy, democratic, democratization $\to$ democr

- Lemmatization:

  democracy, democratic, democratization $\to$ democracy

##### Stop words

Examples:

1. Articles or prepositions
2. Very common words

NLTK (Natural Language Toolkit library for python)

> `sklearn.feature_extraction.text.CountVectorizer`: `max_df`(Maxium frequency of word)

#### N-grams

Help to use local context.

![image-20210203190719053](https://i.loli.net/2021/02/04/rwFMeVC6WmHdBh7.png)

> `sklearn.feature_extraction.text.CountVectorizer`: `Ngram_range, analyzer`

#### Post Processing: TF & IDF

- Term Frequency: $tf = 1/x.sum(axis=1)$
- Inverse Document Frequency: $idf = np.log(x.shape[0] / (x>0).sum(0))$

> `sklearn.feature_extraction.text.TfidfVectorizer`



### Embeddings (~word2vec)

![image-20210203191947746](https://i.loli.net/2021/02/04/YuDpiHTPy4e9x2f.png)

#### Word2vec

- Words: Word2vec, Glove, FastText, etc
- Sentences: Doc2vec, etc



### BOW and w2v Comparison

1. Bag of words

   1. Very large vectors
   2. Meaning of each value in vector is known

2. Word2vec

   1. Relatively small vectors
   2. Values in vector can be interpreted only in some cases
   3. The words with similar meaning often have similar embeddings

   

## Image -> vector

1. Descriptors
2. Train network from scratch
3. Finetuning



# Additional Material and Links

## Feature extraction from text

### Bag of words

- [Feature extraction from text with Sklearn](http://scikit-learn.org/stable/modules/feature_extraction.html)
- [More examples of using Sklearn](https://andhint.github.io/machine-learning/nlp/Feature-Extraction-From-Text/)

### Word2vec

- [Tutorial to Word2vec](https://www.tensorflow.org/tutorials/word2vec)
- [Tutorial to word2vec usage](https://rare-technologies.com/word2vec-tutorial/)
- [Text Classification With Word2Vec](http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/)
- [Introduction to Word Embedding Models with Word2Vec](https://taylorwhitten.github.io/blog/word2vec)



NLP Libraries

- [NLTK](http://www.nltk.org/)
- [TextBlob](https://github.com/sloria/TextBlob)



## Feature extraction from images

### Pretrained models

- [Using pretrained models in Keras](https://keras.io/applications/)
- [Image classification with a pre-trained deep neural network](https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11)

### Finetuning

- [How to Retrain Inception's Final Layer for New Categories in Tensorflow](https://www.tensorflow.org/tutorials/image_retraining)
- [Fine-tuning Deep Learning Models in Keras](https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html)