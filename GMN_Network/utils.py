import mmap
import json

def read_list(read_path):
    seta=list()
    with open(read_path, 'r',encoding='utf-8') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        line = mm.readline()
        while line:
            seta.append(line.decode().strip().lower())
            line = mm.readline()
    mm.close()
    f.close()
    return seta

def write_list(list, write_file):
    fi = open(write_file, "w", encoding="utf-8")
    for key in list:
        fi.write(key)
        fi.write("\n")
    fi.close()

def read_json(pathfile):
    with open(pathfile, 'r', encoding="utf-8") as f:
        data = json.load(f)
    f.close()
    return data

import random
# 按照9:1的比例随机划分为train_ij和valid_ij
def data_split(full_list, ratio=0.9, shuffle=True):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

