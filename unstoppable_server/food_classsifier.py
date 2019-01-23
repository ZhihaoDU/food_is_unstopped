def get_food_information_by_path(img_path, top_num):
    food_label_list = food_classify(img_path, top_num)
    food_information_list = []
    for one_food in food_label_list:
        label = one_food["label"]
        food_info = get_food_information_by_label(label)
        food_information_list.append({**one_food, **food_info})
    return food_information_list


def food_classify(img_path, num):
    return [{"label": 1, "score": "90%"}] * num


def get_food_information_by_label(label):
    food_dict = load_food_dict()
    if food_dict.get(label) is not None:
        return food_dict[label]
    else:
        return None


def load_food_dict():
    food_dict = {1: {"name": "汉堡包"}}
    return food_dict
