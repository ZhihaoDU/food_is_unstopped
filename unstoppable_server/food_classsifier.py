def get_food_information_by_path(img_path):
    food_label = food_classify(img_path)
    food_information = get_food_information_by_label(food_label)
    return food_information


def food_classify(img_path):
    return 1


def get_food_information_by_label(label):
    food_dict = load_food_dict()
    if food_dict.get(label) is not None:
        return food_dict[label]
    else:
        return None


def load_food_dict():
    food_dict = {1: {"name": "汉堡包"}}
    return food_dict
