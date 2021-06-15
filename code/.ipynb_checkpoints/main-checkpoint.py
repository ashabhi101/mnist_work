from __future__ import print_function
import argparse
import sys
import pandas as pd

from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import logging

from loss import SoftCrossEntropyLoss


from torch.utils.tensorboard import SummaryWriter


# Initiate the tensorbaord writer to record logs
writer = SummaryWriter('runs/mnist_work')


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler("module.log"),
                              logging.StreamHandler()])

class Net(nn.Module):
    def __init__(self):
        """Initialize layers for simple CNN arcitecture

        Returns
        ----------
        No return
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of CNN arcitecture

        Parameters
        ----------
            x : {torch.Tensor}
                Input Tensor of shape (batch size, channels, height, width)

        Returns
        ----------
            prediction : torch.Tensor
                Returns prediction tensor of shape (batch size, number of classes)
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(
    args: Dict[str, Any],
    model: Net,
    device: str,
    loss_calc: SoftCrossEntropyLoss,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Adadelta,
    epoch: int,
):
    """Train the model given the set of inputs.

    Parameters
    ----------
        args : {Dict[str, Any]}
            dictionary of arguments for training.
        model : {Net}
            Torch instantiated network class
        device : {str}
            location for training  (cpu or gpu)
        loss_calc : {SoftCrossEntropyLoss}
            custom loss calcuation as defined by you.
        train_loader : {torch.utils.data.DataLoader}
            torch dataloarder
        optimizer : {optim.Adadelta}
            torch optimizer
        epoch : {int}
            current epoch running

    Returns
    ----------
    No return
    """
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        logging.info(data.shape)
        data, target = data.to(device), target.to(device)

        target = target.view([1, -1])
        target_onehot = F.one_hot(torch.as_tensor(target), num_classes = 10).view(-1,10)
        optimizer.zero_grad()
        output = model(data)
        logging.info(output.shape)
        #loss_calc = F.nll_loss(output, target, **kwargs)
        loss = loss_calc(output, target_onehot)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        writer.add_scalar("Loss/train", loss, epoch)

        
        # loss = loss_calc(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break



def test(model: Net, device: str,
loss_calc: SoftCrossEntropyLoss, test_loader: torch.utils.data.DataLoader, epoch):
    """Run test data through the trained model for evaluation.

    Parameters
    ----------
        model : {Net}
            Torch instaciated network class
        device : {str}
            location for training  (cpu or gpu)
        loss_calc : {SoftCrossEntropyLoss}
            custom loss function as defined by you
        test_loader : {torch.utils.data.DataLoader}
            torch data loader

    Returns
    ----------
    No return
    """    
    model.eval()
    test_loss = 0
    correct = 0
    test_res = pd.DataFrame(columns = ['pred', 'actual'])
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.view([1,-1])
            target_onehot = F.one_hot(torch.as_tensor(target), num_classes = 10).view(-1,10)

            output = model(data)
            test_loss += loss_calc(output, target_onehot, reduction="sum").item() 
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            writer.add_scalar("Loss/test", test_loss, epoch) # sum up batch loss
            writer.add_scalar("Accuracy/test", 100.0 * correct / len(test_loader.dataset), epoch)


            for pred_a, target_b in zip(pred.flatten().tolist(),target.flatten().tolist()):
                test_res.loc[len(test_res)+1] = [pred_a,target_b]


    test_res.to_excel('predictions.xlsx', index = False)
    
    logging.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def validation(model: Net, device: str,
loss_calc: SoftCrossEntropyLoss, val_loader: torch.utils.data.DataLoader, epoch):  

    """Run validation data through the trained model for evaluation.

    Parameters
    ----------
        model : {Net}
            Torch instantiated network class
        device : {str}
            location for training  (cpu or gpu)
        loss_calc : {SoftCrossEntropyLoss}
            custom loss function as defined by you
        val_loader : {torch.utils.data.DataLoader}
            torch data loader

    Returns
    ----------
    No return
    """    
    model.eval()
    val_loss = 0
    correct = 0
    val_results = pd.DataFrame(columns = ['pred', 'actual'])
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            target = target.view([1,-1])
            target_onehot = F.one_hot(torch.as_tensor(target), num_classes = 10).view(-1,10)

            output = model(data)
            val_loss += loss_calc(output, target_onehot, reduction="sum").item() 
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            for pred_a, target_b in zip(pred.flatten().tolist(),target.flatten().tolist()):
                val_results.loc[len(val_results)+1] = [pred_a,target_b]


    val_results.to_excel('validation_predictions.xlsx', index = False)


    logging.info(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            val_loss, correct, len(val_loader.dataset), 100.0 * correct / len(val_loader.dataset)
        )
    )



def main():
    """Get user input and execute model training and testing.

    Returns
    ----------
    No return
    """
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=14, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=0.7, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=True, help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # define checkpoint saved path
    ckp_path = "./checkpoints/current_checkpoint.pt"

    # Not using a checkpointed model, running it from scratch

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"batch_size": args.batch_size}
    if use_cuda:
        kwargs.update(
            {"num_workers": 1, "pin_memory": True, "shuffle": True},
        )

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    train_subset, val_subset = torch.utils.data.random_split(
        dataset1, [50000, 10000], generator=torch.Generator().manual_seed(1))
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1) # dataset changed to dataset1 since not using validation set for final run
    val_loader = torch.utils.data.DataLoader(val_subset)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    loss_calc = SoftCrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, loss_calc, train_loader, optimizer, epoch)
        test(model, device, loss_calc, test_loader, epoch)
        #validation(model, device, loss_calc, val_loader, epoch) --> not using the validation set for now
        scheduler.step()


    if args.save_model:
        checkpoint = {
          'state_dict': model.state_dict(),
          'optimizer' : optimizer.state_dict()}
        torch.save(checkpoint, "./checkpoints/mnist_cnn.pt")

writer.close()
if __name__ == "__main__":
    ver = sys.version_info
    assert ver.major == 3 and ver.minor >= 7, "Please run using Python 3.7.0 or higher"
    main()