# parts adapted from https://github.com/nfrn/Multi-Modal-Classification-of-Radiology-Exams
from labelbasedclassification import accuracyMacro, accuracyMicro, precisionMacro, precisionMicro, recallMacro, recallMicro, fbetaMacro, fbetaMicro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import metrics
import matplotlib.colors as colors
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import math
THRESHOLD= 0.5
OPENI=False

def get_classification_metrics(result)
    txts = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

    txts_n = ['No Finding: 17232',
            'Enlarged Cardiomediastinum:   3531',
            'Cardiomegaly:   9446',
            'Airspace Opacity:   7640',
            'Lung Lesion:     626',
            'Edema:   2661',
            'Consolidation:     876',
            'Pneumonia:   1018',
            'Atelectasis:   2768',
            'Pneumothorax:     858',
            'Pleural Effusion:   1260',
            'Pleural Other:       92',
            'Fracture:     365',
            'Support Devices:     266']
    predictions = result[0]
    trueLabels = result[1]
    idx2class = {k: v for k, v in enumerate(txts)}

    print(np.shape(predictions))
    print(np.shape(trueLabels))
    #
    print(type(predictions[0,0]))
    print(type(trueLabels[0,0]))

    confusion_matrixx = confusion_matrix(trueLabels.argmax(axis=1), predictions.argmax(axis=1))
    confusion_matrixx = confusion_matrixx.astype('float') / confusion_matrixx.sum(axis=1)[:, np.newaxis]
    confusion_matrix_df = pd.DataFrame(confusion_matrixx).rename(columns=idx2class, index=idx2class)

    plt.figure()
    ax = sns.heatmap(confusion_matrix_df, annot=False)
    ax.set_yticklabels(txts_n)
    plt.xticks(fontsize=4, rotation=90)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.show()

    print(classification_report(trueLabels.argmax(axis=1), predictions.argmax(axis=1)))

    fpr, tpr, _ = metrics.roc_curve(trueLabels.ravel(), predictions.ravel())
    roc_auc = metrics.auc(fpr, tpr)
    print('\r MICRO val_roc_auc: %s' % (str(round(roc_auc, 4))), end=100 * ' ' + '\n')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(trueLabels.shape[1]):
        fpr[i], tpr[i], _ = metrics.roc_curve(trueLabels[:, i], predictions[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        if(math.isnan(roc_auc[i])):
            roc_auc[i]=0
        print(roc_auc[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(trueLabels.shape[1])]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)

    macro = sum(roc_auc.values())/(trueLabels.shape[1]-1)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(trueLabels.shape[1]):
        fpr[i], tpr[i], _ = metrics.roc_curve(trueLabels[:, i], predictions[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(trueLabels.shape[1])]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(trueLabels.shape[1]):
        if(i==1 and OPENI):
            continue
        # print(interp(all_fpr, fpr[i], tpr[i]))
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    if OPENI:
        mean_tpr /= 13
    else:
        mean_tpr /= trueLabels.shape[1]

    n_classes = trueLabels.shape[1]

    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(trueLabels.ravel(), predictions.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average {0:0.3f}'
                   ''.format(roc_auc["micro"]),
             color='black', linestyle=':', linewidth=3)

    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average {0:0.3f}'
                   ''.format(roc_auc["macro"]),
             color='black', linestyle='-', linewidth=3)

    colors_list = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8',
                   '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#bcf60c', '#fabebe', '#008080', '#e6beff',
                   '#9a6324', '#fffac8', '#800000', '#aaffc3',
                   '#808000', '#ffd8b1', '#000075', '#808080',
                   '#ffffff', '#000000']
    print(colors_list)
    for i, color in zip(range(n_classes), colors_list):
        txt = txts[i]
        if not (math.isnan(roc_auc[i])):
            plt.plot(fpr[i], tpr[i], color=color, lw=1,
                 label='{0} {1:0.2f}'
                       ''.format(txt, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.00])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    #plt.show()
    print('\r MACRO val_roc_auc: %s' % (str(round(macro, 4))), end=100 * ' ' + '\n')

    value = metrics.coverage_error(trueLabels,predictions)
    print('\r MACRO CE: %s' % (str(round(value, 4))), end=100 * ' ' + '\n')
    value = metrics.label_ranking_average_precision_score(trueLabels,predictions)
    print('\r MACRO LRAP: %s' % (str(round(value, 4))), end=100 * ' ' + '\n')

    predictions = (predictions > THRESHOLD).astype(int)

    print("Accuracy Macro:" + str(accuracyMacro(trueLabels, predictions)))
    print("Accuracy Micro:" + str(accuracyMicro(trueLabels, predictions)))
    print("Precision Macro:" + str(precisionMacro(trueLabels, predictions)))
    print("Precision Micro:" + str(precisionMicro(trueLabels, predictions)))
    print("Recall Macro:" + str(recallMacro(trueLabels, predictions)))
    print("Recall Micro:" + str(recallMicro(trueLabels, predictions)))
    print("FBeta Macro:" + str(fbetaMacro(trueLabels, predictions, beta=1)))
    print("FBeta Micro:" + str(fbetaMicro(trueLabels, predictions, beta=1)))


def predict_classification(model_obj, test_loader, device, batch_size = 24, rnn_sizes = (512,1024), num_layers = 2):

    # set model to evaluate model
    model = copy.deepcopy(model_obj)

    model.cuda()

    model.eval()

    y_true = torch.tensor([], dtype=torch.float).cuda()
    all_outputs = torch.tensor([]).cuda()

    states1 = torch.zeros(num_layers, batch_size, rnn_sizes[0])
    states2 = torch.zeros(num_layers, batch_size, rnn_size[2])
    # deactivate autograd engine and reduce memory usage and speed up computations
    with torch.no_grad():
        for i, (x1_batch, x2_batch, label1, label2) in tqdm(enumerate(test_loader)):
            x1_batch = torch.tensor(x1_batch, dtype=torch.long).cuda()
            x2_batch = torch.tensor(x2_batch, dtype=torch.float32).cuda()
            label2 = torch.tensor(label2, dtype=torch.float32).cuda()

            states1 = detach_single(states1)
            states2 = detach_single(states2)

            outputs, states1, states2 = model(x1_batch, x2_batch, states1, states2)
            y_true = torch.cat((y_true, label2), 0)
            all_outputs = torch.cat((all_outputs, outputs), 0)

    y_true = y_true.cpu().numpy()
    y_pred = all_outputs.cpu().numpy()

    array = np.array([y_pred, y_true])
    get_classification_metrics(array)
