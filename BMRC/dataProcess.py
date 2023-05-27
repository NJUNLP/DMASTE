# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4

import pickle
import torch
import os

class dual_sample(object):
    def __init__(self,
                 original_sample,
                 text,
                 forward_querys,
                 forward_answers,
                 backward_querys,
                 backward_answers,
                 sentiment_querys,
                 sentiment_answers):
        self.original_sample = original_sample  #
        self.text = text  #
        self.forward_querys=forward_querys
        self.forward_answers=forward_answers
        self.backward_querys=backward_querys
        self.backward_answers=backward_answers
        self.sentiment_querys=sentiment_querys
        self.sentiment_answers=sentiment_answers


def get_text(lines):
    # Line sample:
    # It is always reliable , never bugged and responds well .####It=O is=O always=O reliable=O ,=O never=O bugged=O and=O responds=T-POS well=O .=O####It=O is=O always=O reliable=O ,=O never=O bugged=O and=O responds=O well=S .=O
    text_list = []
    aspect_list = []
    opinion_list = []
    triplet_data = []
    sentiment_map = {'POS': 0, 'NEG': 1, 'NEU': 2}
    for f in lines:
        temp = f.split("####")
        assert len(temp) == 2
        word_list = temp[0].split()
        ts = eval(temp[1])
        ts = [(t[0], t[1], sentiment_map[t[2]])for t in ts]
        triplet_data.append(ts)
        # aspect_label_list = [t.split("=")[-1] for t in temp[1].split()]
        # opinion_label_list = [t.split("=")[-1] for t in temp[2].split()]

        # aspect_label_list = ['O']
        # assert len(word_list) == len(aspect_label_list) == len(opinion_label_list)
        text_list.append(word_list)
        # aspect_list.append(aspect_label_list)
        # opinion_list.append(opinion_label_list)
    return text_list, aspect_list, opinion_list, triplet_data


def valid_data(triplet, aspect, opinion):
    for t in triplet[0][0]:
        assert aspect[t] != ["O"]
    for t in triplet[0][1]:
        assert opinion[t] != ["O"]


def fusion_dual_triplet(triplet):
    triplet_aspect = []
    triplet_opinion = []
    triplet_sentiment = []
    dual_opinion = []
    dual_aspect = []
    for t in triplet:
        if t[0] not in triplet_aspect:
            triplet_aspect.append(t[0])
            triplet_opinion.append([t[1]])
            triplet_sentiment.append(t[2])
        else:
            idx = triplet_aspect.index(t[0])
            triplet_opinion[idx].append(t[1])
            # assert triplet_sentiment[idx] == sentiment_map[t[2]], f'{triplet_sentiment[idx]} {sentiment_map[t[2]]}'
        if t[1] not in dual_opinion:
            dual_opinion.append(t[1])
            dual_aspect.append([t[0]])
        else:
            idx = dual_opinion.index(t[1])
            dual_aspect[idx].append(t[0])

    return triplet_aspect, triplet_opinion, triplet_sentiment, dual_opinion, dual_aspect


if __name__ == '__main__':
    
    home_path = "../ia-dataset/"
    dataset_name_list = os.listdir(home_path)
    dataset_name_list = [x for x in dataset_name_list if '.' not in x]
    print(dataset_name_list)
    # dataset_type_list = ["train", "test", "dev"]
    for dataset_name in dataset_name_list:
        dataset_dir = os.path.join(home_path, dataset_name)
        dataset_type_list = os.listdir(dataset_dir)
        dataset_type_list = [x.split('.')[0] for x in dataset_type_list]
        for dataset_type in dataset_type_list:
            output_path = "./data/preprocess/" + dataset_name + "_" + dataset_type + "_dual.pt"
            # read triple
            # f = open(home_path + dataset_name + "/" + dataset_name + "_pair/" + dataset_type + "_pair.pkl", "rb")
            # triple_data = pickle.load(f)
            # f.close()
            # read text
            f = open(home_path + dataset_name + "/" + dataset_type + ".txt", "r", encoding="utf-8")
            text_lines = f.readlines()
            f.close()
            # get text
            text_list, _, _, triple_data = get_text(text_lines)
            sample_list = []
            for k in range(len(text_list)):
                triplet = triple_data[k]
                text = text_list[k]
                # valid_data(triplet, aspect_list[k], opinion_list[k])
                triplet_aspect, triplet_opinion, triplet_sentiment, dual_opinion, dual_aspect = fusion_dual_triplet(triplet)
                forward_query_list = []
                backward_query_list = []
                sentiment_query_list = []
                forward_answer_list = []
                backward_answer_list = []
                sentiment_answer_list = []
                forward_query_list.append(["What", "aspects", "?"])
                start = [0] * len(text)
                end = [0] * len(text)
                for ta in triplet_aspect:
                    start[ta[0]] = 1
                    end[ta[-1]] = 1
                forward_answer_list.append([start, end])
                backward_query_list.append(["What", "opinions", "?"])
                start = [0] * len(text)
                end = [0] * len(text)
                for to in dual_opinion:
                    start[to[0]] = 1
                    end[to[-1]] = 1
                backward_answer_list.append([start, end])

                for idx in range(len(triplet_aspect)):
                    ta = triplet_aspect[idx]
                    # opinion query
                    query = ["What", "opinion", "given", "the", "aspect"] + text[ta[0]:ta[-1] + 1] + ["?"]
                    forward_query_list.append(query)
                    start = [0] * len(text)
                    end = [0] * len(text)
                    for to in triplet_opinion[idx]:
                        start[to[0]] = 1
                        end[to[-1]] = 1
                    forward_answer_list.append([start, end])
                    # sentiment query
                    query = ["What", "sentiment", "given", "the", "aspect"] + text[ta[0]:ta[-1] + 1] + ["and", "the",
                                                                                                        "opinion"]
                    for idy in range(len(triplet_opinion[idx]) - 1):
                        to = triplet_opinion[idx][idy]
                        query += text[to[0]:to[-1] + 1] + ["/"]
                    to = triplet_opinion[idx][-1]
                    query += text[to[0]:to[-1] + 1] + ["?"]
                    sentiment_query_list.append(query)
                    sentiment_answer_list.append(triplet_sentiment[idx])
                for idx in range(len(dual_opinion)):
                    ta = dual_opinion[idx]
                    # opinion query
                    query = ["What", "aspect", "does", "the", "opinion"] + text[ta[0]:ta[-1] + 1] + ["describe", "?"]
                    backward_query_list.append(query)
                    start = [0] * len(text)
                    end = [0] * len(text)
                    for to in dual_aspect[idx]:
                        start[to[0]] = 1
                        end[to[-1]] = 1
                    backward_answer_list.append([start, end])

                temp_sample = dual_sample(text_lines[k], text, forward_query_list, forward_answer_list, backward_query_list, backward_answer_list, sentiment_query_list, sentiment_answer_list)
                sample_list.append(temp_sample)
            torch.save(sample_list, output_path)
