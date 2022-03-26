from torch.utils.data import Dataset


class LatentsDataset(Dataset):

	def __init__(self, latents, opts):
		self.latents = latents
		self.opts = opts

	def __len__(self):
		return self.latents.shape[0]

	def __getitem__(self, index):

		return self.latents[index]


class LADataset(Dataset):

	def __init__(self, latents, description, anchors, opts):
		self.latents = latents
		self.anchors = anchors
		self.description = description
		self.opts = opts

	def __len__(self):
		return self.latents.shape[0]

	def __getitem__(self, index):
		self.item = {}
		self.item["latents"] = self.latents[index]
		self.item["anchors"] = ','.join([self.description] + self.anchors[index])

		return self.item


class LAMDataset(Dataset):

	def __init__(self, latents, description, anchors, masks, opts):
		self.latents = latents
		self.anchors = anchors
		self.masks = masks
		self.description = description
		self.opts = opts

	def __len__(self):
		return self.latents.shape[0]

	def __getitem__(self, index):
		self.item = {}
		self.item["latents"] = self.latents[index]
		self.item["anchors"] = ','.join([self.description] + self.anchors[index])
		self.item["masks"] = self.masks[index]

		return self.item
