# adapated from https://github.com/nfrn/Multi-Modal-Classification-of-Radiology-Exams
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

def get_multimodal_data(TRAIN, VAL, TEST, IMG, NF, LA):
    train_df = pd.read_csv(open(TRAIN, 'rU'),
                           encoding='utf-8', engine='c', dtype={'Report': str})
    val_df = pd.read_csv(open(VAL, 'rU'),
                         encoding='utf-8', engine='c', dtype={'Report': str})
    test_df = pd.read_csv(open(TEST, 'rU'),
                          encoding='utf-8', engine='c', dtype={'Report': str})

    print("Load tokenizer")
    with open('tokenizer_reduced.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    print("Preparing train data")
    x1_train = train_df["Indication"].astype(str).values
    x1_train = tokenizer.texts_to_sequences(x1_train)
    x1_train = pad_sequences(x1_train, maxlen=MAX_WORDS_TEXT, padding='post', truncating = 'post')

    x2_train = train_df[IMG].values
    for idx, path in enumerate(x2_train):
        filename = path
        x2_train[idx] = filename

    y1_train = train_df["Impression"].astype(str).values
    y1_train = tokenizer.texts_to_sequences(y1_train)
    y1_train = pad_sequences(y1_train, maxlen=MAX_WORDS_IMPR, padding='post', truncating = 'post')

    y2_train = train_df["Findings"].astype(str).values
    y2_train = tokenizer.texts_to_sequences(y2_train)
    y2_train = pad_sequences(y2_train, maxlen=MAX_WORDS_TEXT, padding='post', truncating = 'post')

    y3_train = train_df[[NF, 'Enlarged Cardiomediastinum', 'Cardiomegaly', LA,
                      'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                      'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']].values
    print("Preparing val data")

    x1_val = val_df["Indication"].astype(str).values
    x1_val = tokenizer.texts_to_sequences(x1_val)
    x1_val = pad_sequences(x1_val, maxlen=MAX_WORDS_TEXT, padding='post', truncating = 'post')

    x2_val = val_df[IMG].values
    for idx, path in enumerate(x2_val):
        filename = path
        x2_val[idx] = filename

    y1_val = val_df["Impression"].astype(str).values
    y1_val = tokenizer.texts_to_sequences(y1_val)
    y1_val = pad_sequences(y1_val, maxlen=MAX_WORDS_IMPR, padding='post', truncating = 'post')

    y2_val = val_df["Findings"].astype(str).values
    y2_val = tokenizer.texts_to_sequences(y2_val)
    y2_val = pad_sequences(y2_val, maxlen=MAX_WORDS_TEXT, padding='post', truncating = 'post')

    y3_val = val_df[[NF, 'Enlarged Cardiomediastinum', 'Cardiomegaly', LA,
                         'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                         'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']].values
    print("Preparing val data")


    x1_test = test_df["Indication"].astype(str).values

    x1_test = tokenizer.texts_to_sequences(x1_test)
    print(len(x1_test))
    print(len(x1_test[0]))
    x1_test = pad_sequences(x1_test, maxlen=MAX_WORDS_TEXT, padding='post', truncating = 'post')
    print(x1_test.shape)
    x2_test = test_df[IMG].values
    for idx, path in enumerate(x2_test):
        filename = path
        x2_test[idx] = filename

    y1_test = test_df["Impression"].astype(str).values
    y1_test = tokenizer.texts_to_sequences(y1_test)
    y1_test = pad_sequences(y1_test, maxlen=MAX_WORDS_IMPR, padding='post', truncating = 'post')

    y2_test = test_df["Findings"].astype(str).values
    y2_test = tokenizer.texts_to_sequences(y2_test)
    y2_test = pad_sequences(y2_test, maxlen=MAX_WORDS_TEXT, padding='post', truncating = 'post')

    y3_test = test_df[[NF, 'Enlarged Cardiomediastinum', 'Cardiomegaly', LA,
                         'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                         'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']].values

    return x1_train, x2_train, y1_train, y3_train, x1_val, x2_val, y1_val, y3_val, x1_test, x2_test, y1_test, y3_test
def getTokenEmbed():
    print("Load tokenizer")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    print("Load embedding_matrix")
    with open('embedding_matrix.pickle', 'rb') as f:
        embedding_matrix = pickle.load(f)

    voc_size = len(tokenizer.word_index) + 1
    return tokenizer, embedding_matrix, voc_size

def getTargetWeights(y):
    weights = np.zeros(y.shape[1])
    for c in range(y.shape[1]):
        weights[c] = np.sum(y[:,c])
    weights = weights/y.shape[0]
    for c in range(y.shape[1]):
        weights[c] = 1 - weights[c]
        weights[c] = weights[c]
    return weights



def prepare_embeddings(t, vocab_size, model, WORD_EMBEDDINGS_SIZE):
    embedding_matrix = np.zeros((vocab_size, WORD_EMBEDDINGS_SIZE))
    for word, i in t.word_index.items():
        embedding_matrix[i] = model.wv[word]
    reverse_word_map = dict(map(reversed, t.word_index.items()))

    print("Saving tokenizer")
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saving embeddings of corpus")
    with open('embedding_matrix.pickle', 'wb') as f:
        pickle.dump(embedding_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

    return embedding_matrix, reverse_word_map
