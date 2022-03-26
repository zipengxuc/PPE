import os
import json
import clip
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

import criteria.clip_loss as clip_loss
from criteria import id_loss
from mapper.datasets.latents_dataset import LatentsDataset
from mapper.styleclip_mapper import StyleCLIPMapper
from mapper.training.ranger import Ranger
from mapper.training import train_utils


class Coach:
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda:0'
        self.opts.device = self.device

        # Initialize network
        self.net = StyleCLIPMapper(self.opts).to(self.device)
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

        # Initialize loss
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss(self.opts).to(self.device).eval()
        if self.opts.clip_lambda > 0:
            self.clip_loss = clip_loss.CLIPLoss(opts).to(self.device)
        if self.opts.latent_l2_lambda > 0:
            self.latent_l2_loss = nn.MSELoss().to(self.device).eval()

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)
        # load max-min CLIP distances for various texts
        with open("./rank/diff_dis.json", 'r') as load_f:
            load_dict = json.load(load_f)
        self.tar_dist = torch.Tensor([load_dict[self.opts.description]]).to(self.device)
        self.anchors = self.opts.anchors.split(',')
        self.distances = torch.Tensor([load_dict[anc] for anc in self.anchors]).to(self.device)  # (10, )
        print(self.distances)
        text_lists = [self.opts.description]+self.anchors
        self.text_combs = torch.cat([clip.tokenize(text_lists)]).cuda()

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    def train(self):
        self.net.train()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                w = batch
                w = w.to(self.device)
                bs = w.size(0)
                with torch.no_grad():
                    x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1)
                    img = self.avg_pool(self.upsample(x))
                w_hat = w + 0.1 * self.net.mapper(w)
                x_hat, w_hat = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
                img_hat = self.avg_pool(self.upsample(x_hat))
                loss, loss_dict = self.calc_loss(w, x, img, w_hat, x_hat, img_hat)
                loss.backward()
                self.optimizer.step()

                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')
                if self.global_step % self.opts.val_interval == 0:
                    with open(os.path.join(self.checkpoint_dir, 'logs.txt'), 'a') as f:
                        f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))
                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('Finished training.')
                    print('Load the best model and save the testing results...')
                    ckpt = torch.load(os.path.join(self.checkpoint_dir, 'best_model.pt'), map_location=self.device)
                    self.net.mapper.load_state_dict(self.get_keys(ckpt, 'mapper'), strict=True)
                    _ = self.validate(best=True)
                    _ = self.validate(best=True,eF=1.5)
                    print("Done!")
                    break

                self.global_step += 1

    def get_keys(self, d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt


    def validate(self, best=False, eF=1.0, save=True):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            if not best and batch_idx > 200:
                break
            if best and batch_idx > 2824-1:  # self.max_iter//16-1 2824-1
                break

            w = batch

            with torch.no_grad():
                w = w.to(self.device).float()
                bs = w.size(0)
                x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=True, truncation=1)
                img = self.avg_pool(self.upsample(x))
                w_hat = w + 0.1 * self.net.mapper(w) * eF
                x_hat, _ = self.net.decoder([w_hat], input_is_latent=True, randomize_noise=True, truncation=1)
                img_hat = self.avg_pool(self.upsample(x_hat))
                loss, cur_loss_dict = self.calc_loss(w, x, img, w_hat, x_hat, img_hat)
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            if save:
                if best and batch_idx <= 200:
                    self.parse_and_log_images(x, x_hat, title='images_val', index=batch_idx, eF=eF)

            # For first step just do sanity test on small amount of data
            if not best and (self.global_step == 0 and batch_idx >= 4):
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')
        if best:
            with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
                f.write('Evaluate - \n{}\n'.format(loss_dict))

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        params = list(self.net.mapper.parameters())
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.latents_train_path:
            train_latents = torch.load(self.opts.latents_train_path)
        else:
            train_latents_z = torch.randn(self.opts.train_dataset_size, 512).cuda()
            train_latents = []
            for b in range(self.opts.train_dataset_size // self.opts.batch_size):
                with torch.no_grad():
                    _, train_latents_b = self.net.decoder([train_latents_z[b: b + self.opts.batch_size]],
                                                          truncation=0.7, truncation_latent=self.net.latent_avg, return_latents=True)
                    train_latents.append(train_latents_b)
            train_latents = torch.cat(train_latents)

        if self.opts.latents_test_path:
            test_latents = torch.load(self.opts.latents_test_path)
        else:
            test_latents_z = torch.randn(self.opts.train_dataset_size, 512).cuda()
            test_latents = []
            for b in range(self.opts.test_dataset_size // self.opts.test_batch_size):
                with torch.no_grad():
                    _, test_latents_b = self.net.decoder([test_latents_z[b: b + self.opts.test_batch_size]],
                                                      truncation=0.7, truncation_latent=self.net.latent_avg, return_latents=True)
                    test_latents.append(test_latents_b)
            test_latents = torch.cat(test_latents)

        train_dataset_celeba = LatentsDataset(latents=train_latents.cpu(),
                                              opts=self.opts)
        test_dataset_celeba = LatentsDataset(latents=test_latents.cpu(),
                                              opts=self.opts)
        train_dataset = train_dataset_celeba
        test_dataset = test_dataset_celeba
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def calc_loss(self, w, x, img, w_hat, x_hat, img_hat):
        loss_dict = {}
        loss = 0.0
        bs = w.size(0)
        # compute CLIP losses
        orig_loss = self.clip_loss(img, self.text_combs).transpose(1,0)
        orig_loss_tar = orig_loss[0].view(-1, bs).transpose(1,0)
        orig_loss_anc = orig_loss[1:].view(-1, bs).transpose(1,0)  # (batch_size, anchors_num)
        mani_loss = self.clip_loss(img_hat, self.text_combs).transpose(1,0)  # (anchors_num+1, batch_size)
        mani_loss_tar = mani_loss[0].view(-1, bs).transpose(1,0)
        mani_loss_anc = mani_loss[1:].view(-1, bs).transpose(1,0)  # (batch_size, anchors_num)
        if self.opts.id_lambda > 0:
            loss_id, sim_improvement = self.id_loss(x_hat, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda
        if self.opts.clip_lambda > 0:
            loss_dict['loss_clip'] = float(mani_loss_tar.mean())
            loss += mani_loss_tar.mean() * self.opts.clip_lambda
        if self.opts.latent_l2_lambda > 0:
            loss_l2_latent = self.latent_l2_loss(w_hat, w)
            loss_dict['loss_l2_latent'] = float(loss_l2_latent)
            loss += loss_l2_latent * self.opts.latent_l2_lambda
        if self.opts.anchor_lambda > 0:
            loss_anchors = torch.square(mani_loss_anc-orig_loss_anc).mean()
            loss += loss_anchors * self.opts.anchor_lambda
            loss_dict['loss_anchors'] = float(loss_anchors)
        # for evaluation
        tar_dis = torch.mean(mani_loss_tar - orig_loss_tar, dim=0)
        anc_dis = torch.mean(mani_loss_anc - orig_loss_anc, dim=0)
        tar_dis_norm = tar_dis / self.tar_dist
        anc_dis_norm = anc_dis / self.distances
        loss_dict["tar_dis"] = float(tar_dis)
        loss_dict["tar_dis_norm"] = float(tar_dis_norm)
        for i in range(len(self.anchors)):
            loss_dict["%s_dis" % self.anchors[i]] = float(anc_dis[i])
            loss_dict["%s_dis_norm" % self.anchors[i]] = float(anc_dis_norm[i])
        de_ind = torch.abs(anc_dis_norm).mean() / torch.abs(tar_dis_norm)
        # print(de_ind)
        loss_dict["indicator"] = float(de_ind)
        loss_dict['loss'] = float(loss)
        return loss, loss_dict

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            #pass
            print(f"step: {self.global_step} \t metric: {prefix}/{key} \t value: {value}")

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, x, x_hat, title, index=None, eF=1.0):
        if index is None:
            path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}.jpg')
        else:
            if eF == 1.0:
                path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}_{str(index).zfill(5)}.jpg')
            else:
                path = os.path.join(self.log_dir, title, f'{str(eF)}_{str(self.global_step).zfill(5)}_{str(index).zfill(5)}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchvision.utils.save_image(torch.cat([x.detach().cpu(), x_hat.detach().cpu()]), path,
                                     normalize=True, scale_each=True, range=(-1, 1), nrow=self.opts.batch_size)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }
        return save_dict
