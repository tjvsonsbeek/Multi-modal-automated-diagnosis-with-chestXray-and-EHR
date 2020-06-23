import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import random
import time
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import copy
import torchvision.models as visionmodels
from torchnlp.nn import Attention
import torch.nn.functional as F

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def isNaN(num):
    return num != num
def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)
class Attention_Net(nn.Module):
    def __init__(self, embedding_matrix, voc_size, embed_size = 200, impression_generation = True, only_image = False, only_text = False):
        super(Attention_Net, self).__init__()
        self.impression_generation = impression_generation
        self.only_image = only_image
        self.only_text = only_text
        self.embedding = nn.Embedding(voc_size, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embed_inverse = nn.Parameter(torch.tensor(np.linalg.pinv(embedding_matrix), dtype=torch.float32))

        self.lstm = nn.GRU(embed_size, 512, bidirectional=True, batch_first=True)
        self.lstm2 = nn.GRU(512 * 2, 1024, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(2048, 2048)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        self.resnet50 = visionmodels.resnet50(pretrained = True)
        self.resnet50filters = nn.Linear(49, 48)
        self.resnet50linear = nn.Linear(2048,2048)
        self.drop = nn.Dropout(p=0.6)
        if not only_image and not only_text:
            self.attention_i0 = Attention(2048)
            self.attention_t0 = Attention(2048)

            self.ff_i_0 = nn.Linear(2048, 2048)
            self.ff_t_0 = nn.Linear(2048, 2048)

            self.dense0t = nn.Linear(2048, 2048)
            self.dense0i = nn.Linear(2048, 2048)

        # downstream block
        self.feature_size = 4096
        if self.only_image:
            self.dense0i = nn.Linear(2048,2048)
            self.feature_size = 2048
        if self.only_text:
            self.dense0t = nn.Linear(2048, 2048)
            self.feature_size = 2048
        if self.impression_generation:
            self.lstm3 = nn.LSTM(self.feature_size, 200, batch_first=True)
            self.out_impression = nn.Linear(200, 200)
        else:
            self.out0 = nn.Linear(self.feature_size, 512)
            self.out1 = nn.Linear(512, 14)

        modules = list(self.resnet50.children())[:-2]  # delete the last fc layer.
        self.resnet50 = nn.Sequential(*modules)
        # set requires_grad to false for cnn backbone
        for param in self.resnet50.parameters():
            param.requires_grad = False

    def forward(self, x, imgs, states1, states2, states3 = None):
        h_embedding = self.embedding(x)
        h_lstm, states1 = self.lstm(h_embedding, states1)
        h_lstm, states2 = self.lstm2(h_lstm, states2)
        h_lstm = self.drop(F.relu(self.linear(self.drop(h_lstm))))

        img_features = self.resnet50(imgs)
        img_features = self.resnet50filters(img_features.view(-1,2048,49))
        img_features = img_features.permute(0,2,1)
        att_img_features = self.resnet50linear(img_features)

        if not self.only_image and not self.only_text:
            # attention module 0
            img_out0, _ = self.attention_i0(att_img_features, h_lstm)
            text_out0, _ = self.attention_t0(h_lstm, att_img_features)

            img_out0 = img_out0 + att_img_features
            text_out0 = text_out0 + h_lstm

            img_out0 = self.drop(self.relu(self.ff_i_0(img_out0)))
            text_out0 = self.drop(self.relu(self.ff_t_0(text_out0)))

        ## HERE BEGINS THE DOWNSTREAM BLOCK

        if self.impression_generation:
            # impression generation
            if self.only_image:
                dense1 = self.drop(F.relu(self.dense0i(att_img_features)))
            elif self.only_text:
                dense1 = self.drop(F.relu(self.dense0t(h_lstm)))
            else:

                img_out = self.drop(F.relu(self.dense0i(img_out0)))

                text_out = self.drop(F.relu(self.dense0t(text_out0)))

                dense1 = torch.cat((text_out, img_out), 2)

            out1, states3 = self.lstm3(dense1, states3)
            impression = self.out_impression(out1)
            impression = torch.matmul(impression, self.embed_inverse)
            impression = impression.permute(0, 2, 1)
            return impression, states1, states2, states3
        else:
            # classification generation
            if self.only_image:
                dense1 = self.drop(F.relu(self.dense0i(torch.sum(att_img_features,1))))
            elif self.only_text:
                dense1 = self.drop(F.relu(self.dense0t(torch.sum(h_lstm,1))))
            else:
                img_out = self.drop(F.relu(self.dense0i(torch.sum(img_out0, 1))))

                text_out = self.drop(F.relu(self.dense0t(torch.sum(text_out0, 1))))

                dense1 = torch.cat((text_out, img_out), 1)

            dense1 = self.drop(F.relu(self.out0(dense1)))
            out = self.out1(dense1)

            outSig = self.sigmoid(out)

            return outSig, states1, states2
    def get_attention_weights(self, x, imgs, states1 = None, states2 = None, text = True):
        h_embedding = self.embedding(x)
        h_lstm, states1 = self.lstm(h_embedding, states1)
        h_lstm, states2 = self.lstm2(h_lstm, states2)

        h_lstm = self.drop(F.relu(self.linear(self.drop(h_lstm))))

        img_features = self.resnet50(imgs)
        img_features = self.resnet50filters(img_features.view(-1, 2048, 49))
        img_features = img_features.permute(0, 2, 1)
        att_img_features = self.resnet50linear(img_features)

        if text:
            _, text_weights = self.attention_t0(h_lstm, att_img_features)
            return text_weights, states1, states2
        else:
            _, image_weights = self.attention_i0(att_img_features, h_lstm)
            return image_weights, states1, states2


def detach_single(state):
    return state.detach().cuda()

def detach_multi(states):
    return [state.detach().cuda() for state in states]
def pytorch_model_run_cv(train_loader, valid_loader, model_obj, model_name, reverse_word_map, tokenizer, clip=True, batch_size = 24, embed_size = 200, num_layers = 2, rnn_sizes = (512,1024), w=None):
    seed_everything()
    model = copy.deepcopy(model_obj)
    model.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=0.00001)
    weights_t = torch.tensor(w[0], dtype=torch.float32).cuda()
    weights_v = torch.tensor(w[1], dtype=torch.float32).cuda()
    ################################################################################################
    ###############################################################################################
    ce_loss = torch.nn.CrossEntropyLoss(reduction = 'none')

    best_valid_loss = float('inf')
    counter = 0
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(n_epochs):
        with tqdm(total=batch_size * len(train_loader)) as epoch_pbar:
            epoch_pbar.set_description(f'Epoch {epoch}')
            start_time = time.time()
            model.train()
            acc_loss = 0.

            states1 = torch.zeros(num_layers, batch_size, rnn_sizes[0])
            states2 = torch.zeros(num_layers, batch_size, rnn_sizes[1])
            if model.impression_generation:
                num_layers = 1
                states3 = (torch.zeros(num_layers, batch_size, embed_size),
                            torch.zeros(num_layers, batch_size, embed_size))
            for i, (x1_batch, x2_batch, y1_batch, y2_batch, weights) in enumerate(train_loader):
                torch.cuda.empty_cache()
                x1_batch = x1_batch.type(torch.long).cuda()
                x2_batch = x2_batch.type(torch.float32).cuda()
                y1_batch = y1_batch.type(torch.long).cuda()
                y2_batch = y2_batch.type(torch.float32).cuda()
                weights = weights.type(torch.float32).cuda()

                states1 = detach_single(states1)
                states2 = detach_single(states2)
                if model.impression_generation: states3 = detach_multi(states3)

                if model.impression_generation:
                    y_pred, states1, states2, states3 = model(x1_batch, x2_batch, states1, states2, states3)
                    loss = ce_loss(y_pred, y1_batch)
                    loss = torch.sum(loss,1)
                    loss = torch.mean(loss * weights)
                else:
                    y_pred, states1, states2 = model(x1_batch, x2_batch, states1, states2)
                    loss = weighted_mse_loss(y_pred, y2_batch,weights_t)

                optimizer.zero_grad()
                loss.backward()

                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                torch.cuda.empty_cache()
                if not isNaN(loss):
                    acc_loss += loss.item()

                avg_loss = acc_loss / (i + 1)
                desc = f'Epoch {epoch} - loss {avg_loss:.4f}'
                epoch_pbar.set_description(desc)
                epoch_pbar.update(x1_batch.shape[0])

        model.eval()

        acc_val_loss = 0.
        with tqdm(total=batch_size * len(valid_loader)) as epoch_pbar:
            epoch_pbar.set_description(f'Epoch {epoch}')
            states1_val = torch.zeros(num_layers, batch_size, rnn_sizes[0])
            states2_val = torch.zeros(num_layers, batch_size, rnn_size[1])
            if model.impression_generation:
                num_layers = 1
                states3_val = (torch.zeros(num_layers, batch_size, embed_size),
                            torch.zeros(num_layers, batch_size, embed_size))
            for i, (x1_batch, x2_batch, y1_batch, y2_batch, weights) in enumerate(valid_loader):
                x1_batch = torch.tensor(x1_batch, dtype=torch.long).cuda()
                x2_batch = torch.tensor(x2_batch, dtype=torch.float32).cuda()
                y1_batch = torch.tensor(y1_batch, dtype=torch.long).cuda()
                y2_batch = torch.tensor(y2_batch, dtype=torch.float32).cuda()
                weights = torch.tensor(weights, dtype=torch.float32).cuda().detach()
                states1_val = detach_single(states1_val)
                states2_val = detach_single(states2_val)
                if model.impression_generation: states3_val = detach_multi(states3_val)
                if model.impression_generation:
                    y_pred, states1_val, states2_val, states3_val = model(x1_batch, x2_batch, states1_val, states2_val, states3_val)
                    y_pred= y_pred.detach()
                    val_loss =ce_loss(y_pred, y1_batch)
                    val_loss = torch.sum(val_loss, 1)
                    val_loss = torch.mean(val_loss * weights)
                else:
                    y_pred, states1_val, states2_val = model(x1_batch, x2_batch, states1_val, states2_val)
                    y_pred = y_pred.detach()
                    val_loss = weighted_mse_loss(y_pred, y2_batch, weights_v)

                acc_val_loss += val_loss
                avg_val_loss = acc_val_loss / (i + 1)
                desc = f'Epoch {epoch} - loss {avg_val_loss:.4f}'
                epoch_pbar.set_description(desc)
                epoch_pbar.update(x1_batch.shape[0])

        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            torch.save(model.state_dict(), model_name)


        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
        if avg_val_loss > avg_loss:
            counter += 1
        if counter == 3:
            break
    return model
