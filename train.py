from src.models import ResMLP
from src.dataset import data_init
from src.utils import train
from src.config import args
import torch

def main(args):
  trainloader, testloader = data_init(args.batch_size)
  model = ResMLP(args.dim,
                args.img_size,
                args.depth,
                args.npatches,
                args.n_classes)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.device(device)

  costFunc = torch.nn.CrossEntropyLoss()
  optimizer =  torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.9)

  train(args.epoch,
        trainloader,
        testloader,
        costFunc,
        model,
        device,
        optimizer)

if __name__ == "__main__":
    main(args)


  