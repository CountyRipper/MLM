#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/25
import pickle
import re
from simcse import SimCSE
import enchant
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from util import fileIO
from stanfordcorenlp import StanfordCoreNLP
import time
import nltk
import numpy as np
from model import input_representation
from stanfordcorenlp import StanfordCoreNLP
import torch
import pyparsing as pp
from transformers import AutoTokenizer, AutoModelForMaskedLM ,AutoModel
from nltk.corpus import stopwords
stopword_dict = set(stopwords.words('english'))
stoplist = ['the','a','/','%','(',')','no','if','an','and','but','is','are','be','were','in','which','of','for','.','!',',','?','that','not','this']
stop_words = stopword_dict.union(stoplist)
punc = "~!§{}¤@#$%^&*()_*/<>,.'[]/?;|\:。॥¦।-"
wnl = nltk.WordNetLemmatizer()
time_start = time.time()
considered_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
d = enchant.Dict("en_US")
from torch.nn import functional as F


def calculate_cos_distance(a,b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    cose = torch.mm(a,b)
    return 1 - cose


def judge_eff(predict):
    flag = 0
    for i in punc:
        if i in predict:
            flag = 1
            break
    return flag


def get_PRF(recall, pre):
    F1 = 0.0
    # P = float(num_c) / float(num_e)
    # R = float(num_c) / float(num_s)
    R = np.mean(recall)
    P = np.mean(pre)
    if (P + R == 0.0):
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)
    return P, R, F1


def is_phrase_in(phrase, text):
    phrase = phrase.lower()
    text = text.lower()

    rule = pp.ZeroOrMore(pp.Keyword(phrase))
    for t, s, e in rule.scanString(text):
        if t:
            return True
    return False


def generate_can(tmp, model, tokenizer, n=50):
    add = []
    #不确定tmp是什么？？？
    #tmp应该是一个名词词性列表，关于这一篇文章
    mask_text = ' '.join(tmp)
    #tokenizer化文本
    inputs = tokenizer(mask_text, return_tensors="pt",truncation=True)
    #cuda化
    inputs["input_ids"] = inputs["input_ids"].cuda()
    inputs["attention_mask"] = inputs["attention_mask"].cuda()
    inputs["token_type_ids"] = inputs["token_type_ids"].cuda()
    #获取masked token的index
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    #获取model预测的结果
    token_logits = model(**inputs).logits
    #获取每个masked 的预测结果向量
    mask_token_logits = token_logits[0, mask_token_index, :]
    #获取前k个可能结果
    top_tokens = torch.topk(mask_token_logits, n, dim=1).indices.tolist()
    #解码获取的token结果
    for token in zip(*top_tokens):
        predict = tokenizer.decode(token)
        #去除停用词
        predict_split = list(filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9,\-\+]', predict)))
        #拼接在一起
        predict = ' '.join(predict_split)
        #判断是否为空
        if predict != '' and predict != ' ':
            # print(predict)
        # predict_split = predict.split()
            #预测长度
            lengh = len(predict_split)
            #获得词性列表
            tag = nltk.pos_tag(predict_split)
            #如果最后一个词在考虑的范围内
            if (tag[-1][-1] in considered_tags):
                # flag = 0
                # for p in predict_split:
                #     if p in stop_words:
                #         flag = 1
                #         break
                # if flag == 1:
                #     continue
                if (len(predict_split) == 1):
                    #judge_eff判断有无标点符号
                    if (judge_eff(predict) == 0):
                        if(d.check(predict)):
                            add.append(predict)
                elif (len(predict_split) == 2):
                    if porter.stem(predict_split[0]) == porter.stem(predict_split[1]):
                        continue
                    else:
                        if (judge_eff(predict) == 0):
                            add.append(predict)
                else:
                    tmp = set()
                    for phrase in predict_split:
                        tmp.add(porter.stem(phrase))
                    b_len = len(list(tmp))
                    if b_len == lengh:
                        add.append(predict)
    return add


def mmr(doc_embedding: np.ndarray,
        word_embeddings: np.ndarray,
        words: List[str],
        top_n: int = 5,
        diversity: float = 0.5) -> List[str]:
    """ Calculate Maximal Marginal Relevance (MMR)
    between candidate keywords and the document.
    MMR considers the similarity of keywords/keyphrases with the
    document, along with the similarity of already selected
    keywords and keyphrases. This results in a selection of keywords
    that maximize their within diversity with respect to the document.
    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_n: The number of keywords/keyhprases to return
        diversity: How diverse the select keywords/keyphrases are.
                   Values between 0 and 1 with 0 being not diverse at all
                   and 1 being most diverse.
    Returns:
         List[str]: The selected keywords/keyphrases
    """

    # Extract similarity within words, and between words and the document
    top_n=min(top_n,len(words))
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    # word_similarity = cosine_similarity(word_embeddings)
    # print("word_similarity:", word_similarity)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]
    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        # target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)
        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)


    return [words[idx] for idx in keywords_idx]


def get_dist_cosine(emb1, emb2):
    vector_a = np.mat(emb1)
    vector_b = np.mat(emb2)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if (denom == 0.0):
        return 0.0
    else:
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim


def if_present_phrase(src_str_tokens, phrase_str_tokens):
    """

    :param src_str_tokens: a list of strings (words) of source text
    :param phrase_str_tokens: a list of strings (words) of a phrase
    :return:
    """
    match_flag = False
    match_pos_idx = -1
    for src_start_idx in range(len(src_str_tokens) - len(phrase_str_tokens) + 1):
        match_flag = True
        # iterate each word in target, if one word does not match, set match=False and break
        for seq_idx, seq_w in enumerate(phrase_str_tokens):
            src_w = src_str_tokens[src_start_idx + seq_idx]
            if src_w != seq_w:
                match_flag = False
                break
        if match_flag:
            match_pos_idx = src_start_idx
            break

    return match_flag



def print_PRF(P, R, F1, N):

    print("\nN=" + str(N), end="\n")
    print("P=" + str(P), end="\n")
    print("R=" + str(R), end="\n")
    print("F1=" + str(F1))
    return 0


def get_all_dist(candidate_embeddings_list, dist_sorted, dist_list):
    '''
    :param candidate_embeddings_list:
    :param text_obj:
    :param dist_list:
    :return: dist_all
    '''

    dist_all = {}
    for i, emb in enumerate(candidate_embeddings_list):
        phrase = dist_sorted[i]
        phrase = phrase.lower()
        tokens = phrase.split()
        phrase_stem = ' '.join(porter.stem(t) for t in tokens)
        if (phrase in dist_all):
            # store the No. and distance
            dist_all[phrase].append(dist_list[i])
        else:
            dist_all[phrase] = []
            dist_all[phrase].append(dist_list[i])
    return dist_all


def get_final_dist(dist_all, method="average"):
    '''
    :param dist_all:
    :param method: "average"
    :return:
    '''

    final_dist = {}

    if method == "average":

        for phrase, dist_list in dist_all.items():
            sum_dist = 0.0
            for dist in dist_list:
                sum_dist += dist
            # if phrase in stop_words:
            #     sum_dist = 0.0
            final_dist[phrase] = sum_dist / float(len(dist_list))
        return final_dist
    elif method == 'max':
        for phrase, dist_list in dist_all.items():
            final_dist[phrase] = max(dist_list)
            # tmp=phrase.split()
            # for word in tmp:
            #     if word in stop_words:
            #         final_dist[phrase] = 0.0
            #         continue

        return final_dist


time_start = time.time()

P = R = F1 = 0.0
num_c_10 = num_c_20 = num_c_50 = num_c_m =0
num_s = 0
lamda = 0.0
recall_10 = []
recall_20 = []
recall_50 = []
recall_m = []
prerecall_5=[]
pre_5=[]
prerecall_10=[]
pre_10=[]
prerecall_o=[]
pre_o=[]
database1 = "Inspec"
database2 ="semeval"
database3 = "nus"
database4="krapivin"

database = database4

if(database == "Inspec"):
    data, labels = fileIO.get_inspec_data()
    with open('inspec_source.pickle', 'rb') as f:
        labels = pickle.load(f)
elif(database == "semeval"):
    data, labels = fileIO.get_semeval_data()
    with open('semeval_source.pickle', 'rb') as f:
      labels = pickle.load(f)
elif(database == "nus"):
    data, labels = fileIO.get_nus_data()
    with open('nus_source.pickle', 'rb') as f:
        labels = pickle.load(f)
else:
    data, labels = fileIO.get_krapivin_data()
    with open('krapivin_source.pickle', 'rb') as f:
      labels = pickle.load(f)



# emb_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-large")
# emb_model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-roberta-large")
# emb_model = SimCSE("../simcse/my-unsup-simcse-roberta-large5")
# emb_model = SimCSE("../simcse/my-unsup-simcse-change-roberta-large5")
emb_model = SimCSE("../simcse/my-unsup-simcse-roberta-large5")
# emb_tokenizer = AutoTokenizer.from_pretrained("../simcse/my-unsup-simcse-change-roberta-large7")
# emb_model = AutoModel.from_pretrained("../simcse/my-unsup-simcse-change-roberta-large7")
# emb_tokenizer = AutoTokenizer.from_pretrained("../simcse/my-unsup-simcse-noev-roberta-large5")
# emb_model = AutoModel.from_pretrained("../simcse/my-unsup-simcse-noev-roberta-large5")
# emb_tokenizer = AutoTokenizer.from_pretrained("../simcse/my-unsup-simcse-roberta-large5")
# emb_model = AutoModel.from_pretrained("../simcse/my-unsup-simcse-roberta-large5")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("../bertbasepad")
# tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
# model = AutoModelForMaskedLM.from_pretrained("../bertlargekp20k")
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking")
model = AutoModelForMaskedLM.from_pretrained("../bert/bertlargewwmkp20k")
# tokenizer = AutoTokenizer.from_pretrained("roberta-large")
# model = AutoModelForMaskedLM.from_pretrained("../bert/robertalargewwmkp20k")
# tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")
# model = AutoModelForMaskedLM.from_pretrained("D:\SIFRank-master\spanbert_large_with_head")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    model = model.cuda()
    # emb_model = emb_model.cuda()
    model.eval()
    # emb_model.eval()
porter = nltk.PorterStemmer()
en_model = StanfordCoreNLP(r'/home/admin02/code/graduate/stanford-corenlp-full-2018-02-27', quiet=True)
# with open('inspec_labelabs.pickle', 'rb') as f:
#     labels = pickle.load(f)
# with open('semeval_labelabs.pickle', 'rb') as f:
#   labels = pickle.load(f)
# with open('nus_labelabs.pickle', 'rb') as f:
#   labels = pickle.load(f)

try:
    total_cannum=0
    len_label=0
    pre_true=0
    abs_true=0
    for key, data in data.items():
        # data 应该是一篇raw text list
        print("key；  ", key)
        # if key not in labels.keys():
        #     print("skip")
        #     continue
        #label是对应文章的label
        lables_stemed = labels[key]
        print(lables_stemed)

        # lables_stemed = []
        # curr_lables[-1] = curr_lables[-1].strip()
        # for lable in curr_lables:
        #     tokens = lable.split()
        #     lables_stemed.append(' '.join(porter.stem(t) for t in tokens))
        # # lables_stemed = list({}.fromkeys(lables_stemed).keys())
        # print("key:", key)
        # pre_len += len(lables_stemed['pre'])
        #sent = "The oxide thickness is calculated using the weight gain and surface area. Artificially changing the surface profile will modify the surface area and calculated oxide thickness. SEM images of samples removed after 111 days oxidation were used to define the change in surface profile length with variation in applied roughness. The profile lengths extracted from the images were then used to modify the length of the sample and therefore the surface area. Table 1 shows the original oxide thicknesses after 111 days oxidation, the modified oxide thicknesses based on the surface profile length and the percentage difference. Results show a maximum decrease in the oxide thickness of 4% when using a surface which accounts for roughness. Comparing the change in oxide thickness between different surface finishes indicates a variation of less than 1%. As such, the impact of the variation in the profile length on the calculated oxide thickness is considered to be insignificant. In addition, if the differences in weight gain were only due to differences in surface area, rougher samples would be expected to demonstrate thicker oxides at the earliest stages of oxidation."
        input = input_representation.InputTextObj(en_model, data)
        #input应该是使用stanfordcoreNLP进行分词
        # inputs = emb_tokenizer(data, padding=True, truncation=True, return_tensors="pt")
        # inputs["input_ids"] = inputs["input_ids"].cuda()
        # inputs["attention_mask"] = inputs["attention_mask"].cuda()
        # inputs["token_type_ids"] = inputs["token_type_ids"].cuda()
        # Get the embeddings
        # with torch.no_grad():
        #     embeddings = emb_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        #     torch.cuda.empty_cache()
        embeddings=emb_model.encode(data)
        #使用simcse获取词嵌入向量
        rank = []
        candidate = []
        tmp = []
        tmp2 = []
        #Mask the words
        for i in range(len(input.keyphrase_candidate)):
            # print("phrase:  ",input.keyphrase_candidate[i][0])
            if(input.keyphrase_candidate[i][1][0] > 512):
                continue
            cflag=0
            tmp = input.tokens[:]
            phrase_len = len(input.keyphrase_candidate[i][0].split())
            if (phrase_len == 1):
                for k in range(input.keyphrase_candidate[i][1][0], input.keyphrase_candidate[i][1][1]):  #对单个词，替换为[MASK]和[MASK][MASK]
                    tmp[k] = tokenizer.mask_token
                    tmp2 = list(tmp[:])
                    tmp2.insert(k + 1, tokenizer.mask_token)
                candidate.extend(generate_can(tmp, model, tokenizer, 15))
                candidate.extend(generate_can(tmp2, model, tokenizer, 40))
            elif (phrase_len >= 2 and phrase_len <= 3):

                for k in range(input.keyphrase_candidate[i][1][0], input.keyphrase_candidate[i][1][1]):
                    tmp[k] = tokenizer.mask_token

                candidate.extend(generate_can(tmp, model, tokenizer, 40))
            # elif (phrase_len >= 4 and phrase_len <= 5):
            #     for k in range(input.keyphrase_candidate[i][1][0], input.keyphrase_candidate[i][1][1] - 2):
            #         tmp = input.tokens[:]
            #         for l in range(k, k + 3):
            #             tmp[l] = tokenizer.mask_token
            #         candidate.extend(generate_can(tmp, model, tokenizer, 40))
                # for k in range(input.keyphrase_candidate[i][1][0], input.keyphrase_candidate[i][1][1] - 1):
                #     tmp = input.tokens[:]
                #     for l in range(k, k + 2):
                #         tmp[l] = tokenizer.mask_token
                #     candidate.extend(generate_can(tmp, model, tokenizer, 40))
            else:
                continue
        pre2 = list({}.fromkeys(candidate).keys())
        dist_sorted = []
        absent_can = set()
        for i, phrase in enumerate(pre2):
            if len(phrase.split()) == 1:
                absent_can.add(phrase)
            else:
                flg = 0
                for w in phrase.split():
                    w=porter.stem(w)
                    if w not in input.tokens_stem:
                        flg = 1
                        break
                if flg == 0:
                    absent_can.add(phrase)  # 添加部分匹配出的absent短语
        total_cannum += len(absent_can)
        dist_sorted = list(absent_can)   
        # for phrase in absent_canl:
        #     tokens = phrase.split()
        #     phrase_stem = [porter.stem(t) for t in tokens]
        #     if if_present_phrase(input.tokens_stem,phrase_stem):
        #         continue
        #     else:
        #         dist_sorted.append(phrase)
        # candidate = emb_tokenizer(dist_sorted, padding=True, truncation=True, return_tensors="pt")
        # candidate["input_ids"] = candidate["input_ids"].cuda()
        # candidate["attention_mask"] = candidate["attention_mask"].cuda()
        # candidate["token_type_ids"] = candidate["token_type_ids"].cuda()
        # with torch.no_grad():
        #     embedding_can = emb_model(**candidate, output_hidden_states=True, return_dict=True).pooler_output
        #     torch.cuda.empty_cache()
        print("present:  ",input.keyphrase_pure)
        print("dist_sorted:   ",dist_sorted)
        # all_can = input.keyphrase_pure + dist_sorted
        all_can = []
        all_can = list(set(input.keyphrase_pure).union(set(dist_sorted)))
        print("all_can:  ",all_can)
        embedding_can=emb_model.encode(all_can)
        dist_list = []
        # embeddings.cpu()
        # embedding_can.cpu()seze(embeddings)
        for i, emb in enumerate(embedding_can):
            dist = torch.cosine_similarity(embeddings, emb,dim=0).item()
            dist_list.append(dist)
        dist_all = get_all_dist(embedding_can, all_can, dist_list)
        dist_final = get_final_dist(dist_all, method='max')
        dist_filer = {}

        final_sorted = sorted(dist_final.items(), key=lambda x: x[1], reverse=True)
        print(final_sorted)
        # dist_list=mmr(embeddings, embedding_can, dist_sorted, 60, 0)
        final_abs = []
        stem_abs = []
        final_pre=[]
        stem_pre =[]
        for tt in final_sorted:
            phrase = tt[0]
            tokens = phrase.split()
            phrase_stem = [porter.stem(t) for t in tokens]
            phrase_nospstem=' '.join(porter.stem(t) for t in tokens)
            if if_present_phrase(input.tokens_stem, phrase_stem)==True:
                if (phrase_nospstem not in stem_pre):
                    stem_pre.append(phrase_nospstem)
                    final_pre.append(phrase)
            else:
                if (phrase_nospstem not in stem_abs):
                    stem_abs.append(phrase_nospstem)
                    final_abs.append(phrase)
        num_c_10 = num_c_20 = num_c_50 = num_c_m = 0
        num_s = len(lables_stemed['abs'])
        print("final_abs:",final_abs[0:50])
        print("final_pre:  ",final_pre)
        j=0
        if(len(lables_stemed['abs']) > 0):
            for tt in stem_abs:
                # tokens = tt[0]
                if (tt in lables_stemed['abs']):
                    if (j < 10):
                        num_c_10 += 1
                        num_c_20 += 1
                        num_c_50 += 1
                        num_c_m += 1
                        print("10以内")

                    elif (j < 20 and j >= 10):
                        num_c_20 += 1
                        num_c_50 += 1
                        num_c_m += 1
                        print("20以内")

                    elif (j < 50 and j >= 20):
                        num_c_50 += 1
                        num_c_m += 1
                        print("50以内")
                    else:
                        num_c_m += 1
                    print("absent_tphrase", tt)
                j += 1
            recal_10 = num_c_10/num_s
            recall_10.append(recal_10)
            recal_20 = num_c_20/num_s
            recall_20.append(recal_20)
            recal_50 = num_c_50/num_s
            recall_50.append(recal_50)
            recal_m = num_c_m/num_s
            recall_m.append(recal_m)

        num_c_5 = num_c_10 = num_c_o = 0
        num_e_5 = num_e_10 = num_e_o = 0
        num_s = len(lables_stemed['pre'])
        j = 0
        if (len(lables_stemed['pre']) > 0):
            for tt in stem_pre:
                # tokens = tt[0]
                if (tt in lables_stemed['pre']):
                    if (j < 5):
                        num_c_5 += 1
                        num_c_10 += 1

                    elif (j < 10 and j >= 5):
                        num_c_10 += 1
                    if (j < num_s):
                        num_c_o += 1
                j += 1
            if (len(stem_abs[0:5]) == 5):
                num_e_5 += 5
            else:
                num_e_5 += len(stem_abs[0:5])

            if (len(stem_abs[0:10]) == 10):
                num_e_10 += 10
            else:
                num_e_10 += len(stem_abs[0:10])

            if (len(stem_abs[0:num_s]) == num_s):
                num_e_o += num_s
            else:
                num_e_o += len(stem_abs[0:num_s])

            prerecal_5 = num_c_5 / num_s
            print("R@5:  ", prerecal_5)
            prerecall_5.append(prerecal_5)
            precion_5 = num_c_5 / num_e_5
            print("P@5:  ", precion_5)
            pre_5.append(precion_5)
            prerecal_10 = num_c_10 / num_s
            print("R@10:  ", prerecal_10)
            prerecall_10.append(prerecal_10)
            precion_10 = num_c_10 / num_e_10
            print("P@10:  ", precion_10)
            pre_10.append(precion_10)
            prerecal_o = num_c_o / num_s
            prerecall_o.append(prerecal_o)
            precion_o = num_c_o / num_e_o
            pre_o.append(precion_o)
        # print("tmp_pretrue:  ", tmp_pretrue, "   tmp_abstrue: " , tmp_abstrue,  "    label_len:" ,  len(lables_stemed))
    #     pre_true += tmp_pretrue
    #     abs_true += tmp_abstrue
    # print("pre_true: ", pre_true)
    # print("abs_true: ", abs_true)
    # print("berttotal_cannum", total_cannum)
    # print("len_label: ", len_label)
    print("absdoc: ",len(recall_10))
    avrecall_10 = np.mean(recall_10)
    print("R@10:  ", avrecall_10)
    avrecall_20 = np.mean(recall_20)
    print("R@20:  ", avrecall_20)
    avrecall_50 = np.mean(recall_50)
    print("R@50:  ", avrecall_50)
    avrecall_m = np.mean(recall_m)
    print("R@m:  ", avrecall_m)
    p, r, f = get_PRF(prerecall_5,pre_5)
    print_PRF(p, r, f, 5)
    p, r, f = get_PRF(prerecall_10, pre_10)
    print_PRF(p, r, f, 10)
    p, r, f = get_PRF(prerecall_o, pre_o)
    print_PRF(p, r, f, 0)
    en_model.close()



except ValueError:
    print("valueerror")
    en_model.close()
time_end = time.time()
print('totally cost', time_end - time_start)