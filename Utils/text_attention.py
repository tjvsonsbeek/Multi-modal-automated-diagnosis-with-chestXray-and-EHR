# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-03-29 16:10:23
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-04-12 09:56:12

## convert the text/attention list to latex code, which will further generates the text heatmap based on attention weights.
import numpy as np
import pickle
latex_special_token = ["!@#$%^&*()"]

def generate(text_list, attention_list, latex_file, color='red', rescale_value = False):
    assert(len(text_list) == len(attention_list))
    if rescale_value:
        attention_list = rescale(attention_list)
    word_num = len(text_list)
    text_list = clean_word(text_list)
    with open(latex_file,'w') as f:
        f.write(r'''\documentclass[varwidth]{standalone}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}'''+'\n')
        string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
        for idx in range(word_num):
            string += "\\colorbox{%s!%s}{"%(color, attention_list[idx])+"\\strut " + text_list[idx]+"} "
        string += "\n}}}"
        f.write(string+'\n')
        f.write(r'''\end{CJK*}
\end{document}''')

def rescale(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_min = np.min(the_array)
    rescale = (the_array - the_min)/(the_max-the_min)*100
    return rescale.tolist()


def clean_word(word_list):
    new_word_list = []
    for word in word_list:
        for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
            if latex_sensitive in word:
                word = word.replace(latex_sensitive, '\\'+latex_sensitive)
        new_word_list.append(word)
    return new_word_list


if __name__ == '__main__':
    ## This is a demo:
    for i in range(24):
        with open("weights_{}.txt".format(i), "rb") as fp:  # Unpickling
            weights = pickle.load(fp)
        with open("indication_{}.txt".format(i), "rb") as fp:  # Unpickling
            text = pickle.load(fp)
        word_num = len(text)
        attention = [w*100 for w in weights]
        color = 'red'
        generate(text, attention, "sample_{}.tex".format(i), color)
