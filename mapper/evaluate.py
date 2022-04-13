"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

from mapper.coach import Coach
from argparse import ArgumentParser


class Options:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', default='output', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--decoder_path', default="./ckpt/stylegan2-ffhq-config-f.pdparams", type=str, help='')
		self.parser.add_argument('--data_path', default="./ckpt/test_faces.npy", type=str, help='')
		self.parser.add_argument('--mapper_path', default="./ckpt/mapper_bangs.pdparams", type=str, help='')


	def parse(self):
		opts = self.parser.parse_args()
		return opts


def main(opts):
	if not os.path.exists(opts.exp_dir):
		# raise Exception('Oops... {} already exists'.format(opts.exp_dir))
		os.makedirs(opts.exp_dir, exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)

	coach = Coach(opts)
	_ = coach.validate(eF=3.0, save=True)


if __name__ == '__main__':
	opts = Options().parse()
	main(opts)
