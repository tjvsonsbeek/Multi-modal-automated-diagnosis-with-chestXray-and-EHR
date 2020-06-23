import numpy as np
import pandas as pd
from collections import Counter
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm

def get_section(report, section_name, section_status = False):
    if report.find(section_name+':')!=-1:

        words = report.split(section_name+':')[1]
        uppercasecounter = 0
        for id, char in enumerate(words):
            if char.isupper():
                uppercasecounter+=1
            else:
                uppercasecounter=0

            if uppercasecounter>5:
                words = words[:(id-uppercasecounter)]
        section = words.replace('\n', ' ')
        section_status = True
    else:
        section = ''
    return section, section_status
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text
def process_sections(sections_text, sections_names):
    impression = ''
    indication = ''
    findings = ''
    d = {' ': '', '_': '', '\n': '', '\t': ''}
    if replace_all(sections_text[0],d) != '':
        impression = 'IMPRESSION: {}'.format(sections_text[0])
    for s_id in range(len(sections_text[4:])):
        if replace_all(sections_text[s_id+4],d)!='':
            indication += '{}: {} \n'.format(sections_names[s_id+4], sections_text[4+s_id])
    findings = 'FINDINGS: {}'.format(sections_text[3])
    if impression == '' or indication == '':
        impression = ''
        indication = ''
    return impression, indication, findings


def extract_txt_file(path):
    with open(path, 'r') as file:
        report = file.read()


    sections = ['IMPRESSION', 'CONCLUSION','PROVISIONAL FINDINGS IMPRESSION (PFI)', 'FINDINGS','INDICATION', 'HISTORY','TECHNIQUE','STUDY','EXAM']

    section_texts = []
    section_exists = []
    # indication finder
    for s in sections:
        text, text_exist = get_section(report, s)
        section_texts.append(text)
        section_exists.append(text_exist)
    # impression postprocessing
    if not section_exists[0] and section_exists[1]:
        section_exists[0] = True
        section_texts[0] = section_texts[1]
    elif not section_exists[0] and section_exists[2]:
        section_exists[0] = True
        section_texts[0] = section_texts[2]
    # OPTINAL: if no impression present: take findings as impression
    # if not section_exists[0] and section_exists[3]:
    #     section_exists[0] = True
    #     section_texts[0] = section_texts[3]
    impression, indication, findings = process_sections(section_texts, sections)

    return impression, indication, findings

def mergeMIMIC():

    result = pd.read_csv('/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv')
    df = pd.read_csv('/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv')
    df = df.loc[(df['ViewPosition'] == 'PA') | (df['ViewPosition'] == 'AP')]
    new_result = pd.DataFrame(columns=np.append(result.columns.values, np.array(['Path', 'Path_compr', 'Indication', 'Impression', 'Findings'])), index=range(df["dicom_id"].values.shape[0]))
    print(new_result)
    paths = df["dicom_id"].values.copy()
    empty = 0
    c_nf = 0
    for i in tqdm(range(paths.shape[0])):
        p_compr = '/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/compressed_images224/files/' + 'p{}/p{}/s{}/'.format(
            str(df['subject_id'].values[i])[:2], df['subject_id'].values[i], df['study_id'].values[i]) + paths[
                     i] + '.jpg'
        p_txt = '/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr/2.0.0/files/' + 'p{}/p{}/s{}.txt'.format(
            str(df['subject_id'].values[i])[:2], df['subject_id'].values[i], df['study_id'].values[i])

        p = '/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'+'p{}/p{}/s{}/'.format(str(df['subject_id'].values[i])[:2], df['subject_id'].values[i],df['study_id'].values[i])+paths[i]+'.jpg'

        result_index = result.index[(result['subject_id'] == df['subject_id'].values[i]) & (result['study_id'] == df['study_id'].values[i])]
        impression, indication, findings = extract_txt_file(p_txt)

        try:
            if impression != '':
                class_values = result.loc[result_index].values[0]
                class_values = np.nan_to_num(class_values)
                class_values = np.where(class_values==-1.0, 0.0, class_values)
                class_values = np.where(class_values == -9.0, 0.0, class_values)

                if np.count_nonzero(class_values[2:])==0:
                    class_values[10] = 1.0

                input = list(class_values) + [p, p_compr, indication, impression, findings]

                new_result.iloc[i] = input
                c_nf+=1

        except:
            print(input)
            print("SHITSHIT")
            empty+=1
            print("empty: {}".format(empty))

    print(c_nf)
    new_result.to_csv("/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/total_multi_mimic_0706_textgen.csv")


def stratify():
    df = pd.read_csv("/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/improved_multi_mimic_0709_text_gen.csv", usecols=['Path_compr','Indication', 'Impression', 'Findings', 'No Finding', 'Enlarged '
                                                                             'Cardiomediastinum',
                                           'Cardiomegaly', 'Lung Opacity','Lung Lesion',
                                           'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                                           'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                                           'Fracture', 'Support Devices'])

    totalX = df[['Path_compr','Indication', 'Impression', 'Findings']].values
    totalY = df[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices']].values

    print(totalX.shape)
    print(totalY.shape)

    totalX = np.expand_dims(totalX, axis=1)

    print("PRE ITERATIVE")
    X_train, y_train, X_test, y_test = iterative_train_test_split(totalX, totalY, 0.2)

    print("COMBINATION")
    df = pd.DataFrame({
        'train': Counter(
            str(combination) for row in get_combination_wise_output_matrix(y_train, order=2)
            for
            combination in row),
        'test': Counter(
            str(combination) for row in get_combination_wise_output_matrix(y_test, order=2) for
            combination in row)
    }).T.fillna(0.0)
    print(df.to_string())

    X_train = np.squeeze(X_train, axis=1)
    X_test = np.squeeze(X_test, axis=1)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)


    print("WRITING Train")

    dfTotal2 = pd.DataFrame(columns=['Path_compr','Indication', 'Impression', 'Findings', 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices'])
    print(dfTotal2.shape)
    dfTotal2[['Path_compr','Indication', 'Impression', 'Findings']] = pd.DataFrame(X_train)
    dfTotal2[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices']] = y_train

    with open("/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/train_multi2_v3.csv", mode='w', newline='\n') as f:
        dfTotal2.to_csv(f, sep=",", float_format='%.2f', index=False, line_terminator='\n',
                    encoding='utf-8')


    print("WRITING Test")

    dfTotal2 = pd.DataFrame(columns=['Path_compr','Indication', 'Impression', 'Findings', 'No Finding', 'Enlarged Cardiomediastinum',
                                     'Cardiomegaly', 'Lung Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices'])
    dfTotal2[['Path_compr','Indication', 'Impression', 'Findings']] = pd.DataFrame(X_test)
    dfTotal2[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices']] = y_test
    with open("/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/test_multi_v3.csv", mode='w', newline='\n') as f:
        dfTotal2.to_csv(f, sep=",", float_format='%.2f', index=False, line_terminator='\n',
                    encoding='utf-8')
def stratify_val():
    df = pd.read_csv("/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/train_multi2_v3.csv", usecols=['Path_compr', 'Indication', 'Impression', 'Findings', 'No Finding', 'Enlarged '
                                                                                                             'Cardiomediastinum',
                                                                             'Cardiomegaly', 'Lung Opacity',
                                                                             'Lung Lesion',
                                                                             'Edema', 'Consolidation', 'Pneumonia',
                                                                             'Atelectasis',
                                                                             'Pneumothorax', 'Pleural Effusion',
                                                                             'Pleural Other',
                                                                             'Fracture', 'Support Devices'])

    totalX = df[['Path_compr', 'Indication', 'Impression', 'Findings']].values
    totalY = df[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']].values

    print(totalX.shape)
    print(totalY.shape)

    totalX = np.expand_dims(totalX, axis=1)

    print("PRE ITERATIVE")
    X_train, y_train, X_test, y_test = iterative_train_test_split(totalX, totalY, 0.2)

    print("COMBINATION")
    df = pd.DataFrame({
        'train': Counter(
            str(combination) for row in get_combination_wise_output_matrix(y_train, order=2)
            for
            combination in row),
        'test': Counter(
            str(combination) for row in get_combination_wise_output_matrix(y_test, order=2) for
            combination in row)
    }).T.fillna(0.0)
    print(df.to_string())

    X_train = np.squeeze(X_train, axis=1)
    X_test = np.squeeze(X_test, axis=1)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    print("WRITING Train")

    dfTotal2 = pd.DataFrame(
        columns=['Path_compr', 'Indication', 'Impression', 'Findings', 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'])
    print(dfTotal2.shape)
    dfTotal2[['Path_compr', 'Indication', 'Impression', 'Findings']] = pd.DataFrame(X_train)
    dfTotal2[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
              'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']] = y_train

    with open("/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/train_multi_v3.csv", mode='w', newline='\n') as f:
        dfTotal2.to_csv(f, sep=",", float_format='%.2f', index=False, line_terminator='\n',
                        encoding='utf-8')

    print("WRITING Test")

    dfTotal2 = pd.DataFrame(columns=['Path_compr', 'Indication', 'Impression', 'Findings', 'No Finding', 'Enlarged Cardiomediastinum',
                                     'Cardiomegaly', 'Lung Opacity',
                                     'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                                     'Pneumothorax',
                                     'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'])
    dfTotal2[['Path_compr', 'Indication', 'Impression', 'Findings']] = pd.DataFrame(X_test)
    dfTotal2[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
              'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']] = y_test
    with open("/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/val_multi_v3.csv", mode='w', newline='\n') as f:
        dfTotal2.to_csv(f, sep=",", float_format='%.2f', index=False, line_terminator='\n',
                        encoding='utf-8')
if __name__ == '__main__':

    mergeMIMIC()
    stratify()
    stratify_val()
