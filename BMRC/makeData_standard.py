# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4

import torch
import pickle

from dataProcess import get_text


def make_standard(home_path, dataset_name, dataset_type):
    # read triple
    f = open(home_path + dataset_name + "/" + dataset_type + ".txt", "r", encoding="utf-8")
    text_lines = f.readlines()
    f.close()
    # get text
    _, _, _, triple_data = get_text(text_lines)

    standard_list = []

    for triplet in triple_data:

        aspect_temp = []
        opinion_temp = []
        pair_temp = []
        triplet_temp = []
        asp_pol_temp = []
        for temp_t in triplet:
            triplet_temp.append([temp_t[0][0], temp_t[0][-1], temp_t[1][0], temp_t[1][-1], temp_t[2]])
            ap = [temp_t[0][0], temp_t[0][-1], temp_t[2]]
            if ap not in asp_pol_temp:
                asp_pol_temp.append(ap)
            a = [temp_t[0][0], temp_t[0][-1]]
            if a not in aspect_temp:
                aspect_temp.append(a)
            o = [temp_t[1][0], temp_t[1][-1]]
            if o not in opinion_temp:
                opinion_temp.append(o)
            p = [temp_t[0][0], temp_t[0][-1], temp_t[1][0], temp_t[1][-1]]
            if p not in pair_temp:
                pair_temp.append(p)

        standard_list.append({'asp_target': aspect_temp, 'opi_target': opinion_temp, 'asp_opi_target': pair_temp,
                     'asp_pol_target': asp_pol_temp, 'triplet': triplet_temp})

    return standard_list


if __name__ == '__main__':
    
    home_path = "../ia-dataset/"
    sources = ['electronics', 'beauty', 'fashion', 'home', '14res', '15res', '16res', '14lap', 'all']
    targets = ['book', 'grocery', 'office', 'pet', 'toy']
    for dataset_name in sources + targets:
        output_path = "./data/preprocess/" + dataset_name + "_standard.pt"
        dev_standard = make_standard(home_path, dataset_name, 'dev')
        test_standard = make_standard(home_path, dataset_name, 'test')
        torch.save({'dev': dev_standard, 'test': test_standard}, output_path)
        # else:
        #     test_standard = make_standard(home_path, dataset_name, 'test')
        #     torch.save({'dev': None, 'test': test_standard}, output_path)
