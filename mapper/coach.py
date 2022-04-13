import os
import json
import paddle
import numpy as np
from latent_mappers import LevelsMapper
from stylegan import StyleGANv2Generator
import cv2

class Coach:
    def __init__(self, opts):
        self.opts = opts
        # Initialize network
        self.mapper = LevelsMapper()
        self.decoder = StyleGANv2Generator(1024, 512, 8)
        self.init_weight()

        # Initialize dataset
        self.test_face_data = np.load(self.opts.data_path)

    def init_weight(self):
        checkpoint = paddle.load(self.opts.decoder_path)
        self.decoder.set_state_dict(checkpoint)
        self.decoder.eval()
        checkpoint = paddle.load(self.opts.mapper_path)
        self.mapper.set_state_dict(checkpoint)
        self.mapper.eval()


    def validate(self, eF=1.0, save=True):
        for idx in range(100):
            w = self.test_face_data[idx:idx+1]
            w = paddle.to_tensor(w)

            with paddle.no_grad():
                x, _ = self.decoder([w], input_is_latent=True, randomize_noise=True, truncation=1)
                w_hat = w + 0.1 * self.mapper(w) * eF
                x_hat, _ = self.decoder([w_hat], input_is_latent=True, randomize_noise=True, truncation=1)

            x_out = x.numpy()

            if save:
                self.parse_and_log_images(x, x_hat, title='images_val', index=idx, eF=eF)
            break

        return 0


    def parse_and_log_images(self, x, x_hat, title, index=None, eF=1.0):
        x_out = paddle.concat([x, x_hat], axis=3)
        x_out = paddle.transpose(x_out[0], [1, 2, 0]) * 0.5 + 0.5
        x_out = x_out.numpy()[:, :, ::-1]
        

        path = os.path.join(self.opts.exp_dir, title, f'res_{str(eF)}__{str(index).zfill(5)}.jpg')
        print(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, x_out * 255)
