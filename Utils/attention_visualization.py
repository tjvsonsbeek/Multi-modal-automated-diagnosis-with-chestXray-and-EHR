import numpy as np
import copy
import torch
import pickle
from tqdm import tqdm
import cv2
def detach_single(state):
    return state.detach().cuda()
def visualize_text_attention_weights(model_obj, test_loader, device, reverse_word_map, num_layers = 2, batch_size =24, rnn_weights = (512, 1024], max_num_words = 48):
    # generate and save text attention weights as .txt file for one batch. Multimodal input is also saved for reference

    model = copy.deepcopy(model_obj)
    model.cuda()
    # set model to evaluate model
    model.eval()
    states1 = torch.zeros(num_layers, batch_size, rnn_weights[0])
    states2 = torch.zeros(num_layers, batch_size, rnn_weights[1])

    with torch.no_grad():
        # for efficieny dataloader should contain one batch
        for i, (x1_batch, x2_batch, temp1, temp2, ws) in tqdm(enumerate(test_loader)):
            x1_batch = torch.tensor(x1_batch, dtype=torch.long).cuda()
            x2_batch = torch.tensor(x2_batch, dtype=torch.float32).cuda()
            states1 = detach_single(states1)
            states2 = detach_single(states2)


            attention_weights, states1, states2 = model.get_attention_weights(x1_batch, x2_batch, states1, states2)
            text = x1_batch.detach().cpu().numpy()
            img = x2_batch.detach().cpu().numpy()
            print(x1_batch.shape)
            print(attention_weights.size())
            attention_weights = attention_weights.detach().cpu().numpy()
            attention_weights = (attention_weights - np.amin(attention_weights))/(np.amax(attention_weights)-np.amin(attention_weights))



            for b in range(batch_size):
                indication = []
                weights = []
                for w in range(max_num_words-1):
                    if reverse_word_map.get(text[b,w+1]):
                        indication.append(reverse_word_map.get(text[b,w+1]))
                        weights.append(np.mean(attention_weights[b,:,w+1]))
                weights = np.array(weights)
                weights = (weights - np.amin(weights))/(np.amax(weights)-np.amin(weights))
                with open("weights_{}.txt".format(b), "wb") as fp:  # Pickling
                    pickle.dump(weights, fp)
                with open("indication_{}.txt".format(b), "wb") as fp:  # Pickling
                    pickle.dump(indication, fp)

                img2 = img[b,:,:,:]*255
                img3 = np.zeros((224,224,3))
                img3[:, :, 0] = img2[0, :, :]
                img3[:, :, 1] = img2[0, :, :]
                img3[:, :, 2] = img2[2, :, :]
                cv2.imwrite("im_{}.png".format(b), img3)
