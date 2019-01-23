import os
import sys
import main
import numpy as np


def food_classify(img_path, num):
    label, score = main.test(img_path)
    ret = []
    for li, si in zip(label, score):
        ret.append({"label": int(li), "score": "%.2f" % (float(si)*100)})
    return ret
