import os
import sys
import time
import clip
import json
import torch
from torch.utils.data import DataLoader

sys.path.append(".")
sys.path.append("..")

from mapper.datasets.latents_dataset import LatentsDataset
from mapper.styleclip_mapper import StyleCLIPMapper


class RankAndClassify:
    def __init__(self, tar_text, cand_texts, opts):
        self.text_lists = tar_text
        self.tar_texts = torch.cat([clip.tokenize(tar_text)]).cuda()
        self.cand_texts = torch.cat([clip.tokenize(cand_texts)]).cuda()
        self.opts = opts
        self.device = 'cuda:0'
        self.net = StyleCLIPMapper(self.opts).to(self.device)
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        train_latents = torch.load(self.opts.latents_train_path)
        self.train_dataset = LatentsDataset(latents=train_latents.cpu(),
                                              opts=self.opts)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           drop_last=True)
        self.rank_dicts = {}


    def run(self):
        cla_lists = []
        for batch_idx, batch in enumerate(self.train_dataloader):
            if batch_idx % 1000:
                print(batch_idx)
            w = batch
            w = w.to(self.device)
            with torch.no_grad():
                x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1)
                img = self.avg_pool(self.upsample(x))
                # rank
                logits_per_image, _ = self.clip_model(img, self.tar_texts)
                dist = 1 - logits_per_image / 100  # [batch_size, num]
                for j in range(8):
                    self.rank_dicts[batch_idx*8+j] = dist[j].cpu().numpy().tolist()
                # classify
                logits_per_image, _ = self.clip_model(img, self.cand_texts)
                probs = logits_per_image.softmax(dim=-1)
                probs = torch.argmax(probs, dim=-1).cpu().numpy()
                for j in range(8):
                    if probs[j] != 0:
                        cla_lists.append(batch_idx*8+j)
            if batch_idx > 3020:
                break
        rank_lists = sorted(self.rank_dicts.items(), key=lambda kv: (kv[1], kv[0]))
        rank_lists_idx = []
        for i in range(len(rank_lists)):
            rank_lists_idx.append(rank_lists[i][0])
        for item in cla_lists:
            if item in rank_lists_idx:
                rank_lists_idx.remove(item)
        new_dicts = []
        for idx in rank_lists_idx:
            new_dicts.append((idx, self.rank_dicts[idx]))

        with open("./preprocess/%s.json" % self.text_lists, "w") as f:
            json.dump(new_dicts, f)
        print("Done!!!!")


cand_dicts = {"gender": ["male", "female"],
              "hair color": ["black hair", "blond hair", "brown hair", "grey hair", "red hair"],
              "hair length": ["long hair", "short hair"],
              "hair style": ["curly hair", "straight hair", "bald", "wavy hair", "fringe hair", "receding hairline"],
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
              "beard": ["goatee", "mustache", "no beard", "sideburns", "5 o'clock shadow"],
              "rosy cheeks": ["a face with rosy cheeks", "a face without rosy cheeks"],
              "bags under eyes": ["a face with bags under eyes", "a face without bags under eyes"],
              "with lipstick": ["a face with lipstick", "a face without lipstick"],
              "smiling": ["a face with smile", "a face without smile"],
              "with glasses": ["a face with glasses", "a face without glasses"],
              "with bangs": ["a face with bangs", "a face without bangs"],
              "high cheekbones": ["a face with high cheekbones", "a face without high cheekbones"],
              "double chin": ["a face with double chin", "a face without double chin"],
              "with earrings": ["a face with earrings", "a face without earrings"],
              "with makeup": ["a face with makeup", "a face without makeup"],
              "chubby": ["a chubby face", "a not chubby face"],
              "with wrinkles": ["a face with wrinkles", "a face without wrinkles"]}

cla_dicts = {"gender": ["male", "female"],
              "hair color": ["black hair", "blond hair", "brown hair", "grey hair", "red hair"],
              "hair length": ["long hair", "short hair"],
              "hair style": ["curly hair", "straight hair", "bald", "wavy hair", "fringe hair", "receding hairline"],
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
               "rosy cheeks", "bags under eyes", "high cheekbones", "chubby"]

for item in others_list:
    cla_dicts[item] = [item]


if __name__ == '__main__':
    from mapper.options.train_options import TrainOptions
    opt = TrainOptions()
    opt.parser.add_argument('--cmd', type=str)
    opts = opt.parse()
    opts.batch_size = 8
    cmd = opts.cmd

    rev_dicts = {}
    for key in cla_dicts.keys():
        newk = cla_dicts[key]
        for value in newk:
            rev_dicts[value] = key
    candidates = cand_dicts[rev_dicts[cmd]]

    ranker = RankAndClassify(cmd, candidates, opts)
    ranker.run()

