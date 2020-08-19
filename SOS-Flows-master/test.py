from models import *
#from maf import *
from training import *
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

d=6

#torch.manual_seed(1)

num_hidden = 100
num_blocks = 1
num_inputs = d
K = 1
M = 2
lr = 0.0001
bs = 100
epochs = 10


target = torch.distributions.MultivariateNormal(torch.ones(d)*2,torch.eye(d)*3)
Xtrain = target.sample(torch.Size([150000]))
Xval = target.sample(torch.Size([10000]))
Xtest = target.sample(torch.Size([10000]))

train_data, val_data, test_data = make_datasets(Xtrain.numpy(), Xval.numpy(), Xtest.numpy())
print(train_data)
plt.plot(xtrain)
plt.show()

#dataset = getattr(datasets, 'POWER')()
#train_data, val_data, test_data = make_datasets(dataset.trn.x, dataset.val.x, dataset.tst.x)

train_loader, val_loader, test_loader = make_loaders(train_data, val_data, test_data, bs, 1000)

model, optimizer = build_model(num_inputs, num_hidden, K, M, num_blocks, lr, device=device)
#model, optimizer = build_maf(num_inputs, num_hidden, num_blocks, lr, device)
best_model_forward, test_loss_forward = train(model, optimizer, train_loader, val_loader, test_loader,
                                              epochs, device, 500)

