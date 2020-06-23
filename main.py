import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import random
import time
from tqdm import tqdm
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from torch.utils.data import Dataset

from Utils.load_data import get_multimodal_data, prepare_embeddings, \
    getTargetWeights
from Utils.attention_visualization import visualize_text_attention_weights
from Models.multimodal_attention import Attention_Net, pytorch_model_run_cv
from Evaluation.evaluate_impression_generation import predict_impression
from Evaluation.evaluate_classification import predict_classification

embed_size = 200
maxlen = 48
batch_size = 40
n_epochs = 200
SEED = 10
debug = 0

def get_data_paths(MIMIC):
    if MIMIC:
        TXT = 'Report'
        IMG = 'Path_compr'
        NF = 'No Finding'
        LA = 'Lung Opacity'
        TRAIN = "../train.csv"
        TEST = "../test.csv"
        VAL = "../val.csv"
    else:
        TXT = 'Report'
        IMG = 'Img'
        NF = 'No findings'
        LA = 'Airspace Opacity'
        TRAIN = "../train.csv"
        TEST = "../test.csv"
        VAL = "../val.csv"
    return TXT, IMG, NF, LA, TRAIN, TEST, VAL
def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class MultimodalDataset(Dataset):
    def __init__(self, T, X, y1, y2, w=None):
        self.T = T
        self.X = X
        self.y1 = y1
        self.y2 = y2
        self.w = w

    def __len__(self):
        return self.T.shape[0]

    def __getitem__(self, idx):
        img = img_to_array(load_img(self.X[idx], color_mode="rgb", target_size=(224, 224))) / 255.
        img = np.moveaxis(img, 2, 0)

        if self.w is not None:
            weights = self.w[np.argmax(self.y2[idx,:])]
            return self.T[idx], img, self.y1[idx], self.y2[idx,:], weights
        else:
            return self.T[idx], img, self.y1[idx], self.y2[idx,:]




if __name__ == '__main__':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        MIMIC = True
        impression_generation = False
        TXT, IMG, NF, LA, TRAIN, TEST, VAL = get_data_paths(MIMIC)
        print('Loading data...')

        x1_train, x2_train, y1_train, y2_train, x1_val, x2_val, y1_val, y2_val, x1_test, x2_test, y1_test, y2_test = get_multimodal_data(TRAIN, VAL, TEST,
                                                                                                           IMG, NF, LA)
        w_train = getTargetWeights(y2_train)
        w_val = getTargetWeights(y2_val)
        w_test = getTargetWeights(y2_test)
        print('Loading Tokenizer, Embedding...')
        tokenizer, embedding_matrix, voc_size = getTokenEmbed()

        reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))


        pre_load_model_name = ".pt"
        if MIMIC:
            if impression_generation:
                # text generation
                model_name = ".pt"
            else:
                # classification
                model_name = ".pt"
        else:
            model_name = ".pt"
        train_loader = torch.utils.data.DataLoader(dataset=MultimodalDataset(x1_train, x2_train, y1_train, y2_train, w_train),
                                                   batch_size=batch_size, drop_last = True,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=MultimodalDataset(x1_val, x2_val, y1_val, y2_val, w_val),
                                                 batch_size=batch_size, drop_last = True,
                                                 shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=MultimodalDataset(x1_test[:240], x2_test[:240], y1_test[:240], y2_test[:240]),
                                                  batch_size=batch_size, drop_last = True,
                                                  shuffle=True)


        model_name = ".pt"

        model = Attention_Net(embedding_matrix=embedding_matrix, voc_size=voc_size,
                              impression_generation=impression_generation, only_image=False, only_text=False)

        model.load_state_dict(torch.load(model_name))
        model = pytorch_model_run_cv(train_loader, val_loader, model, model_name, reverse_word_map=reverse_word_map,
                                     tokenizer=tokenizer, clip=False, w=(w_train, w_val))
        visualize_text_attention_weights(model, train_loader, device, reverse_word_map)

        model.load_state_dict(torch.load(model_name))
        if impression_generation:
            predict_impression(model, test_loader, device, reverse_word_map)
        else:
            predict_classification(model, test_loader, device)
