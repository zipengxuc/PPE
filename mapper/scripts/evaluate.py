"""
This file runs the main training/val loop
"""
import os
import json
import torch
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

from mapper.options.train_options import TrainOptions
from mapper.training.coach import Coach


def main(opts):
	if not os.path.exists(opts.exp_dir):
		# raise Exception('Oops... {} already exists'.format(opts.exp_dir))
		os.makedirs(opts.exp_dir, exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)

	coach = Coach(opts)
	ckpt = torch.load(os.path.join(coach.checkpoint_dir, 'best_model.pt'), map_location=coach.device)
	coach.net.mapper.load_state_dict(coach.get_keys(ckpt, 'mapper'), strict=True)
	# self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
	print("eval 1.5")
	_ = coach.validate(best=True, eF=0.7, save=True)
	_ = coach.validate(best=True, eF=1.5, save=True)
	print("eval 2.0")
	_ = coach.validate(best=True, eF=2.0, save=True)
	_ = coach.validate(best=True, eF=2.5, save=True)
	print("eval 3.0")
	_ = coach.validate(best=True, eF=3.0, save=True)
	_ = coach.validate(best=True, eF=3.5, save=True)
	print("eval 4.0")
	_ = coach.validate(best=True, eF=4.0, save=True)


if __name__ == '__main__':
	opts = TrainOptions().parse()
	main(opts)
