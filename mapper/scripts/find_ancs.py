import os
import sys
import time
import clip
import json
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import operator
sys.path.append(".")
sys.path.append("..")

import criteria.clip_loss as clip_loss
from criteria import id_loss
from mapper.datasets.latents_dataset import LatentsDataset
from mapper.styleclip_mapper import StyleCLIPMapper
from mapper.training.ranger import Ranger
from mapper.training import train_utils


class FindAncs:
    def __init__(self, comd, latents, cand_texts, opts):
        self.comd = comd
        self.text_lists = cand_texts
        self.cand_texts = torch.cat([clip.tokenize(cand_texts)]).cuda()
        self.nums = len(cand_texts)
        self.opts = opts
        self.bs = opts.batch_size
        self.device = 'cuda:0'
        self.net = StyleCLIPMapper(self.opts).to(self.device)
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)
        self.clip_loss = clip_loss.CLIPLoss(opts).to(self.device)
        self.latents = latents
        self.max_iter = len(self.latents)
        self.train_dataloader = DataLoader(self.latents,
                                           batch_size=self.opts.batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           drop_last=False)
        self.rank_dicts = {}

    def find(self):
        mem_counter = torch.zeros(1, self.nums).to(self.device)
        for batch_idx, batch in enumerate(self.train_dataloader):
            if batch_idx % 1000:
                print(batch_idx)
            w = batch
            w = w.to(self.device)
            with torch.no_grad():
                x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1)
                img = self.avg_pool(self.upsample(x))
                clip_dists = self.clip_loss(img, self.cand_texts)  # [batch_size, num]
                mem_counter += torch.sum(clip_dists, 0)

            if batch_idx >= self.max_iter//self.bs-1:  # self.max_iter//16-1
                break
        sorted_, indices = torch.sort(mem_counter, descending=True)
        sorted_ = sorted_.cpu().numpy()
        indices = indices.cpu().numpy()
        ancs_list = []
        for i in range(self.nums):
            print(sorted_[0][i])
            print(self.text_lists[indices[0][i]])
            ancs_list.append(self.text_lists[indices[0][i]])
        print(ancs_list[::-1])
        full_orig = ["square face", "with makeup", "with earrings", "grey eyes", "short eyebrows", "thin eyebrows",
                     "bags under eyes", "dark eyebrows", "oval face", "high eyebrows", "thin nose", "round eyebrows",
                     "round face", "arched eyebrows", "small nose", "grey hair", "smiling", "small eyes",
                     "long eyebrows", "flat nose", "round eyes", "thick eyebrows", "green eyes", "pinched nose",
                     "narrow eyes", "straight eyebrows", "short hair", "wide eyes", "white skin", "brown eyes",
                     "with lipstick", "big nose", "black eyes", "blue eyes", "thick nose", "short nose", "with bangs",
                     "receding hairline", "close mouth", "hooked nose", "long nose", "big eyes", "5 o'clock shadow",
                     "long hair", "wavy hair", "blond hair", "closed eyes", "open eyes", "with wrinkles", "male",
                     "sideburns", "high cheekbones", "rosy cheeks", "yellow skin", "long face", "open mouth",
                     "pointy face", "no beard", "brown hair", "with glasses", "small mouth", "pointed nose", "bald",
                     "straight hair", "curly hair", "goatee", "big mouth", "female", "black skin", "mustache",
                     "black hair", "red hair"]
        full = clean_full_ranks(cand_dicts, full_orig, exc=rev_dicts[command])
        unique = ancs_list[::-1]
        full_dict = dict(zip(full, range(1, len(full) + 1)))
        uniq_dict = dict(zip(unique, range(1, len(unique) + 1)))
        rr_dict = {}
        for i in range(len(unique)):
            r_full = full_dict[unique[i]] if full_dict[unique[i]] < 40 else 40
            rr_dict[unique[i]] = uniq_dict[unique[i]] / r_full
        srr_dict = dict(sorted(rr_dict.items(), key=operator.itemgetter(1)))
        print(srr_dict.keys())
        ancs = []
        for key in srr_dict.keys():
            cla = rev_dicts[key]
            if classes[cla] == 0:
                ancs.append(key)
                classes[cla] = 1
        print(ancs)
        print(ancs[:20])

        with open("./preprocess/atrs %s.json" % self.comd, "w") as f:
            json.dump(ancs, f)
        print("Done!!!!")


cand_dicts = {"gender": ["male", "female"],
              "hair color": ["black hair", "blond hair", "brown hair", "grey hair", "red hair"],
              "hair length": ["long hair", "short hair"],
              "hair style": ["curly hair", "straight hair", "bald", "wavy hair", "receding hairline"],
              "eye color": ["blue eyes", "brown eyes", "black eyes", "grey eyes", "green eyes"],
              "eye status": ["open eyes", "closed eyes"],
              "eye shape": ["narrow eyes", "wide eyes", "big eyes", "small eyes", "round eyes"],
              "nose shape": ["big nose", "long nose","pointed nose", "small nose", "hooked nose",
                             "short nose", "thick nose", "thin nose", "pinched nose", "flat nose"],
              "face shape": ["pointy face", "round face", "square face", "oval face", "long face"],
              "skin color": ["white skin", "black skin", "yellow skin"],
              "mouth status": ["open mouth", "close mouth"],
              "mouth size": ["big mouth", "small mouth"],
              "eyebrows": ["round eyebrows", "high eyebrows", "arched eyebrows", "long eyebrows", "thick eyebrows",
                           "dark eyebrows", "straight eyebrows", "thin eyebrows", "short eyebrows"],
              "beard": ["goatee", "mustache", "no beard", "sideburns", "5 o'clock shadow"]}

others_list = ["with earrings",
               "with makeup",
               "smiling",
               "with lipstick",
               "with wrinkles",
               "with glasses",
               "with bangs",
               "rosy cheeks", "bags under eyes", "high cheekbones"]

for item in others_list:
    cand_dicts[item] = [item]

rev_dicts = {}
for key in cand_dicts.keys():
    newk = cand_dicts[key]
    for value in newk:
        rev_dicts[value] = key

classes = {}
for key in cand_dicts.keys():
    classes[key] = 0


def make_cand_lists(cand_dicts, exc):
    cand_list = []
    for key in cand_dicts.keys():
        if key != exc:
            for item in cand_dicts[key]:
                cand_list.append(item)
    return cand_list


def clean_full_ranks(cand_dicts, full_orig, exc):
    full = full_orig
    for key in cand_dicts[exc]:
        print(key)
        full.remove(key)
    return full


if __name__ == '__main__':
    from mapper.options.train_options import TrainOptions
    opts = TrainOptions().parse()
    opt.parser.add_argument('--cmd', type=str)
    opts = opt.parse()
    opts.batch_size = 8
    command = opts.cmd
    with open("preprocess/%s.json" % command,'r') as load_f:
        load_dict = json.load(load_f)[:100]
    idx_list = []
    for item in load_dict:
        idx_list.append(item[0])
    train_latents = torch.load(opts.latents_train_path).cuda()
    select_latents = torch.index_select(train_latents, 0, torch.tensor(idx_list).cuda())
    train_dataset_celeba = LatentsDataset(latents=select_latents.cpu(),
                                          opts=opts)
    cand_list = make_cand_lists(cand_dicts, exc=rev_dicts[command])
    finder = FindAncs(command, train_dataset_celeba, cand_list, opts)
    finder.find()
