import challenge.challenge_p1 as p1
import pytest
from torch import tensor
import torch

def test_CustomLoss():
	"""
	Function to test if the loss function is working as defined
	"""
	input_tensor = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
	target = torch.tensor([[1,0], [1,0]])
	loss = p1.SoftCrossEntropyLoss()
	assert torch.equal((loss(input = input_tensor, target = target, reduction = 'sum')*10**4).round()/10**4, tensor(0.5556))

