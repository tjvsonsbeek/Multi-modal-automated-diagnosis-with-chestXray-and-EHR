import numpy as np
import torch
import copy
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from copy import deepcopy
rouge = Rouge()

def print_sentences_and_bleu(s_true, s_pred, reverse_word_map):
    sentence = []
    sentence_true = [[]]
    for word in range(s_pred.shape[2]):

        encoded_word = reverse_word_map.get(np.argmax(s_pred[0, :, word]))
        if encoded_word:
            sentence.append(encoded_word)

        true_word = reverse_word_map.get(s_true[0, word])
        if true_word:
            sentence_true[0].append(true_word)
    if sentence != '':
        print("================")
        print(sentence_bleu(sentence_true, sentence))
        print(sentence)
        print(sentence_true[0])
        print("------------------")
def get_bleu_from_output(s_pred, s_true, reverse_word_map, order, weights):
    bleu_score_tot = 0
    print(s_pred.shape)
    for batch in range(s_pred.shape[0]):
        sentence = []
        sentence_true = [[]]
        for word in range(s_pred.shape[2]):

            encoded_word = reverse_word_map.get(np.argmax(s_pred[batch, :, word]))
            if encoded_word:
                sentence.append(encoded_word)

            true_word = reverse_word_map.get(s_true[batch, word])
            if true_word:
                sentence_true[0].append(true_word)


        bleu_score = sentence_bleu(sentence_true, sentence, weights = order)
        bleu_score_tot += bleu_score/s_pred.shape[0]

    return bleu_score_tot

def get_rouge_meteor_from_output(s_pred, s_true, reverse_word_map, order, weights):
    r_score_tot = 0
    k_score_tot = 0
    for batch in range(s_pred.shape[0]):
        sentence = ''
        sentence_true = ['']
        for word in range(s_pred.shape[2]):

            encoded_word = reverse_word_map.get(np.argmax(s_pred[batch, :, word]))
            if encoded_word:
                sentence +=' '+ encoded_word

            true_word = reverse_word_map.get(s_true[batch, word])
            if true_word:
                sentence_true[0] += " "+ true_word
        r_score = rouge.get_scores(sentence_true[0], sentence)
        k_score = meteor_score(sentence_true, sentence, 4)

        r_score = r_score[0]['rouge-1']['f']
        r_score_tot += r_score/s_pred.shape[0]
        k_score_tot += k_score/s_pred.shape[0]

    return r_score_tot, k_score_tot

def detach_single(state):
    state.detach().cuda()

def detach_multi(states):
    return [state.detach().cuda() for state in states]

def predict_impression(model_obj, test_loader, device, reverse_word_map, batch_size = 24, embed_size = 200, rnn_sizes = (512,1024), num_layers = 2):
    # set model to evaluate model
    model = copy.deepcopy(model_obj)

    model.cuda()

    model.eval()

    y_true = torch.tensor([], dtype=torch.float).cuda()
    all_outputs = torch.tensor([]).cuda()
    counter = 0

    bleu_1_imp = 0
    bleu_2_imp = 0
    bleu_3_imp = 0
    bleu_4_imp = 0

    rouge_imp = 0
    meteor_imp = 0

    states1 = torch.zeros(num_layers, batch_size, rnn_sizes[0])
    states2 = torch.zeros(num_layers, batch_size, rnn_sizes[1])

    states3 = (torch.zeros(1, batch_size, embed_size),
                   torch.zeros(1, batch_size, embed_size))

    with torch.no_grad():
        for i, (x1_batch, x2_batch, label1, label2) in tqdm(enumerate(test_loader)):
            counter+=1
            x1_batch = torch.tensor(x1_batch, dtype=torch.long).cuda()
            x2_batch = torch.tensor(x2_batch, dtype=torch.float32).cuda()
            label1 = torch.tensor(label1, dtype=torch.long).cuda()
            label2 = label2.numpy()
            states1 = detach_single(states1)
            states2 = detach_single(states2)
            states3 = detach_multi(states3)

            output1, states1, states2, states3 = model(x1_batch, x2_batch, states1, states2, states3)

            bleu_1_imp += get_bleu_from_output(output1.detach().cpu().numpy(), label1.cpu().numpy(), reverse_word_map, [1], weights = label2)
            bleu_2_imp += get_bleu_from_output(output1.detach().cpu().numpy(), label1.cpu().numpy(), reverse_word_map, (0.5,0.5),  weights = label2)
            bleu_3_imp += get_bleu_from_output(output1.detach().cpu().numpy(), label1.cpu().numpy(), reverse_word_map, (0.33,0.33,0.33),  weights = label2)
            bleu_4_imp += get_bleu_from_output(output1.detach().cpu().numpy(), label1.cpu().numpy(), reverse_word_map, (0.25,0.25,0.25,0.25),  weights = label2)


            rouge, meteor = get_rouge_meteor_from_output(output1.detach().cpu().numpy(), label1.cpu().numpy(), reverse_word_map, [1], weights = label2)
            rouge_imp += rouge
            meteor_imp += meteor
            print("IMPRESSION:")
            print("BLEU 1: {}".format(bleu_1_imp/counter))
            print("BLEU 2: {}".format(bleu_2_imp/counter))
            print("BLEU 3: {}".format(bleu_3_imp/counter))
            print("BLEU 4: {}".format(bleu_4_imp/counter))
            print("ROUGE : {}".format(rouge_imp/counter))
            print("METEOR: {}".format(meteor_imp/counter))
