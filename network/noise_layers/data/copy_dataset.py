import os
import shutil
from os.path import join as osp

from tqdm import tqdm


source_data = "/home/dangpb1/Research/datasets/PhoneCraft2/Blur/images"
save_data = "/home/dangpb1/Research/datasets/PhoneCraft2/full/blur"

# source_files = sorted(os.listdir(source_data))
source_files = [
    "20230331_225011",
    "20230331_235418",
    "20230401_134357",
    "20230401_134529",
    "20230401_142918",
    "20230401_142932",
]
# '20230331_225011', '20230331_235418', '20230401_134357', '20230401_134529', '20230401_142918', '20230401_142932'
# source
for cate in source_files:
    cate_f = sorted(os.listdir(osp(source_data, cate)))
    for each in tqdm(cate_f):
        shutil.copyfile(osp(source_data, cate, each), osp(save_data, cate + "_" + each))
