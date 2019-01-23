import os
import sys
sys.path.append("..")
import main

import numpy as np

def get_food_information_by_path(img_path, top_num):
    food_label_list = food_classify(img_path, top_num)
    food_information_list = []
    for one_food in food_label_list:
        label = one_food["label"]
        food_info = get_food_information_by_label(label)
        food_information_list.append({**one_food, **food_info})
    return food_information_list


def food_classify(img_path, num):
    label, score = main.test(img_path)
    ret = []
    for li, si in zip(label, score):
        ret.append({"label": int(li), "score": "%.2f" % (float(si)*100)})
    return ret#[{"label": 1, "score": "90%"}] * num


def get_food_information_by_label(label):
    if global_food_dict.get(label) is not None:
        return global_food_dict[label]
    else:
        return None


def load_food_dict():
    f = open("food_name.csv")
    raw_list = f.readlines()
    food_dict = {}
    for i in range(len(raw_list)):
        one_food = raw_list[i].replace('\n', '')
        food_dict[i] = {"name": one_food.split(',')[-1],
                        "en_name": one_food.split(',')[0]}
    return food_dict

global_food_dict = load_food_dict()
