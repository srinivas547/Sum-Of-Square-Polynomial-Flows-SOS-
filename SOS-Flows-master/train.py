import argparse
from utils import *

#
#   Code for masking, datasets and MAF implementation taken from https://github.com/ikostrikov/pytorch-flows
#   We thank Ilya Kostrikov for the initial code which we adapted for this implementation.
#

TEST_BATCH_SIZE = 100

if __name__ == '__main__':

    if sys.version_info < (3, 6):
        print('Sorry, this code might need Python 3.6 or higher')

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Flows')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='input batch size for training (default: 100)')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1000,
        help='input batch size for testing (NOTE: currently disabled and fixed at 100)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='number of epochs to train (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument(
        '--dataset',
        default='MOONS',
        help='POWER | GAS | HEPMASS | MINIBONE | BSDS300 | MOONS')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--num-blocks',
        type=int,
        default=5,
        help='number of invertible blocks (default: 5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1000,
        help='how many batches to wait before logging training status')
    parser.add_argument(
        '--K',
        type=int,
        default=5,
        help='number of polynomials for SOS')
    parser.add_argument(
        '--R',
        type=int,
        default=1,
        help='degree of polynomials for SOS')
    parser.add_argument(
        '--name',
        type=str,
        default='',
        help='run name')
    parser.add_argument(
        '--mode',
        type=str,
        default='direct',
        help='mode')
    parser.add_argument(
        '--flow',
        type=str,
        default='SOS',
        help='flow type (SOS or MAF)',
        choices=['SOS', 'MAF'])

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    assert args.dataset in ['POWER', 'GAS', 'HEPMASS', 'MINIBONE', 'BSDS300', 'MOONS']
    dataset = getattr(datasets, args.dataset)()

    train_dataset, valid_dataset, test_dataset = make_datasets(dataset.trn.x, dataset.val.x, dataset.tst.x)
    train_loader, valid_loader, test_loader = make_loaders(train_dataset, valid_dataset, test_dataset,
                                                           args.batch_size, TEST_BATCH_SIZE, **kwargs)

    num_inputs = dataset.n_dims
    num_hidden = {
        'POWER': 100,
        'GAS': 100,
        'HEPMASS': 512,
        'MINIBOONE': 512,
        'BSDS300': 512,
        'MOONS': 64
    }[args.dataset]

    if args.flow == 'SOS':
        model, optimizer = build_model(num_inputs, num_hidden, args.K, args.R, args.num_blocks, args.lr, device=device)
    else:
        model, optimizer = build_maf(num_inputs, num_hidden, args.num_blocks, args.lr, device=device)
    best_model_forward, test_loss_forward = train(model, optimizer, train_loader, valid_loader, test_loader,
                                                  args.epochs, device, args.log_interval)

    name = args.name if len(args.name) > 0 else default_name()
    path = MODEL_DIR + name

    save_dict = {
        'model': best_model_forward,
        #'optim': optimizer,
        'args': args
    }
    torch.save(save_dict, path)
