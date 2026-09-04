import torch
import torch.nn as nn
import torch.nn.functional as F

class lossAV(nn.Module):
	def __init__(self):
		super(lossAV, self).__init__()
		self.criterion = nn.BCELoss()
		self.FC        = nn.Linear(128, 2)
		
	def forward(self, x, labels = None, r = 1):	
		x = x.squeeze(1)
		x = self.FC(x)
		if labels == None:
			predScore = x[:,1]
			predScore = predScore.t()
			predScore = predScore.view(-1).detach().cpu().numpy()
			return predScore
		else:
			x1 = x / r
			x1 = F.softmax(x1, dim = -1)[:,1]
			nloss = self.criterion(x1, labels.float())
			predScore = F.softmax(x, dim = -1)
			predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
			correctNum = (predLabel == labels).sum().float()
			return nloss, predScore, predLabel, correctNum


class lossV(nn.Module):
	def __init__(self):
		super(lossV, self).__init__()
		self.criterion = nn.BCELoss()
		self.FC        = nn.Linear(128, 2)

	def logits(self, x):
		x = x.squeeze(1)
		return self.FC(x)

	def forward_from_logits(self, logits, labels, r = 1):
		return self.frame_losses_from_logits(logits, labels, r).mean()

	def frame_losses_from_logits(self, logits, labels, r = 1):
		"""Return per-frame visual BCE for M03 reliability ranking."""
		x = F.softmax(logits / r, dim = -1)
		return F.binary_cross_entropy(
			x[:,1], labels.float(), reduction='none'
		)

	def forward(self, x, labels, r = 1):	
		return self.forward_from_logits(self.logits(x), labels, r)
