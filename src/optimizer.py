import torch

def set_optimizer(paramslist, optimizer, lr, wd, mom):

  optimizers = {
    'adam': torch.optim.Adam(
      params=paramslist,
      lr=lr,
      weight_decay=wd,
    ),

    'adagrad': torch.optim.Adagrad(
      params=paramslist,
      lr=lr,
      weight_decay=wd,
    ),

    'momentum': torch.optim.SGD(
      params=paramslist,
      lr=lr,
      momentum=mom,
      weight_decay=wd,
    ),

    'rmsprop': torch.optim.RMSprop(
      params=paramslist,
      lr=lr,
      momentum=mom,
      weight_decay=wd,
    ),
  }

  try:
    return optimizers[optimizer]
  except ValueError:
    return ValueError('Unknown optimizer.')
    