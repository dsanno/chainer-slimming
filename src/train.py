import argparse
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import time


import chainer
from chainer import link
from chainer import links as L
from chainer import optimizers
from chainer import serializers

import net
import trainer


CIFAR10_MEAN = np.asarray([[[0.49140099]], [[0.48215911]], [[0.44653094]]], dtype=np.float32)


def _parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 dataset trainer')
    parser.add_argument('structure_path', type=str,
                        help='Structure JSON file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU device ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Input model file path')
    parser.add_argument('--batch-size', '-b', type=int, default=128,
                        help='Mini batch size')
    parser.add_argument('--prefix', '-p', type=str, default=None,
                        help='Prefix of model parameter files')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Training epoch')
    parser.add_argument('--save-epoch', type=int, default=0,
                        help='Epoch interval to save model parameter file.')
    parser.add_argument('--lr-decay-epoch', type=str, default='150,225',
                        help='Epoch interval to decay learning rate')
    parser.add_argument('--lr-shape', type=str, default='multistep', choices=['multistep', 'cosine'],
                        help='Learning rate annealing function, multistep or cosine')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--lambda-value', type=float, default=0.0001,
                        help='Factor for regularization gamma of batch normalization')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='Optimizer name')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate for SGD')
    parser.add_argument('--alpha', type=float, default=0.001,
                        help='Initial alpha for Adam')
    parser.add_argument('--no-valid-data', action='store_true',
                        help='Do not use validation data')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    return parser.parse_args()


def _subtract_mean(data):
    image, label = data
    return image - CIFAR10_MEAN, label


def _transform(data):
    x, label = data
    width, height = x.shape[-2:]
    x = x - CIFAR10_MEAN
    image = np.zeros_like(x)
    offset = np.random.randint(-4, 5, size=(2,))
    mirror = np.random.randint(2)
    top, left = offset
    left = max(0, left)
    top = max(0, top)
    right = min(width, left + width)
    bottom = min(height, top + height)
    if mirror > 0:
        x = x[:,:,::-1]
    image[:,height-bottom:height-top,width-right:width-left] = x[:,top:bottom,left:right]
    return image, label


def _add_hook_to_gamma(chain, hook):
    for child in chain.children():
        if isinstance(child, link.Chain):
            _add_hook_to_gamma(child, hook)
        elif isinstance(child, L.BatchNormalization):
            child.gamma.update_rule.add_hook(hook)


def main():
    args = _parse_args()

    np.random.seed(args.seed)
    if args.prefix is None:
        model_prefix = os.path.basename(args.structure_path)
        model_prefix = os.path.splitext(model_prefix)[0]
    else:
        model_prefix = args.prefix
    log_file_path = os.path.join('model', '{}_log.csv'.format(model_prefix))
    lr_decay_epoch = map(int, args.lr_decay_epoch.split(','))

    print('loading dataset...')
    train_data, test_data = chainer.datasets.get_cifar10()
    if args.no_valid_data:
        valid_data = None
    else:
        train_data, valid_data = chainer.datasets.split_dataset_random(train_data, 45000)
    train_data = chainer.datasets.TransformDataset(train_data, _transform)
    test_data = chainer.datasets.TransformDataset(test_data, _subtract_mean)
    if valid_data is not None:
        valid_data = chainer.datasets.TransformDataset(valid_data, _subtract_mean)

    print('start training')
    with open(args.structure_path) as f:
        output_sizes = json.load(f)
    cifar_net = net.VGG(output_sizes)
    if args.model is not None:
        serializers.load_npz(args.model, cifar_net)

    if args.optimizer == 'sgd':
        optimizer = optimizers.MomentumSGD(lr=args.lr)
    else:
        optimizer = optimizers.Adam(alpha=args.alpha)
    optimizer.setup(cifar_net)
    if args.lambda_value > 0:
        _add_hook_to_gamma(cifar_net, chainer.optimizer.Lasso(args.lambda_value))
    if args.weight_decay > 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
    cifar_trainer = trainer.CifarTrainer(cifar_net, optimizer, args.epoch, args.batch_size, args.gpu, lr_shape=args.lr_shape, lr_decay=lr_decay_epoch)

    state = {'best_valid_error': 100, 'best_test_error': 100, 'clock': time.clock()}
    def on_epoch_done(epoch, n, o, loss, acc, valid_loss, valid_acc, test_loss, test_acc, test_time):
        error = 100 * (1 - acc)
        print('epoch {} done'.format(epoch))
        print('train loss: {} error: {}'.format(loss, error))
        if valid_loss is not None:
            valid_error = 100 * (1 - valid_acc)
            print('valid loss: {} error: {}'.format(valid_loss, valid_error))
        else:
            valid_error = None
        if test_loss is not None:
            test_error = 100 * (1 - test_acc)
            print('test  loss: {} error: {}'.format(test_loss, test_error))
            print('test time: {}s'.format(test_time))
        else:
            test_error = None
        if valid_loss is not None and valid_error < state['best_valid_error']:
            save_path = os.path.join('model', '{}.model'.format(model_prefix))
            serializers.save_npz(save_path, n)
            save_path = os.path.join('model', '{}.state'.format(model_prefix))
            serializers.save_npz(save_path, o)
            state['best_valid_error'] = valid_error
            state['best_test_error'] = test_error
        elif valid_loss is None:
            save_path = os.path.join('model', '{}.model'.format(model_prefix))
            serializers.save_npz(save_path, n)
            save_path = os.path.join('model', '{}.state'.format(model_prefix))
            serializers.save_npz(save_path, o)
            state['best_test_error'] = test_error
        if args.save_epoch > 0 and (epoch + 1) % args.save_epoch == 0:
            save_path = os.path.join('model', '{}_{}.model'.format(model_prefix, epoch + 1))
            serializers.save_npz(save_path, n)
            save_path = os.path.join('model', '{}_{}.state'.format(model_prefix, epoch + 1))
            serializers.save_npz(save_path, o)
        clock = time.clock()
        print('elapsed time: {}'.format(clock - state['clock']))
        state['clock'] = clock
        with open(log_file_path, 'a') as f:
            f.write('{},{},{},{},{},{},{}\n'.format(epoch, loss, error, valid_loss, valid_error, test_loss, test_error))

    with open(log_file_path, 'w') as f:
        f.write('epoch,train loss,train acc,valid loss,valid acc,test loss,test acc\n')
    cifar_trainer.fit(train_data, valid_data, test_data, on_epoch_done)

    print('best test error: {}'.format(state['best_test_error']))

    train_loss, train_acc, test_loss, test_acc = np.loadtxt(log_file_path, delimiter=',', skiprows=1, usecols=[1, 2, 5, 6], unpack=True)
    epoch = len(train_loss)
    xs = np.arange(epoch, dtype=np.int32) + 1
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(xs, train_loss, label='train loss', c='blue')
    ax.plot(xs, test_loss, label='test loss', c='red')
    ax.set_xlim((1, epoch))
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend(loc='upper right')
    save_path = os.path.join('model', '{}_loss.png'.format(model_prefix))
    plt.savefig(save_path, bbox_inches='tight')

    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(xs, train_acc, label='train error', c='blue')
    ax.plot(xs, test_acc, label='test error', c='red')
    ax.set_xlim([1, epoch])
    ax.set_xlabel('epoch')
    ax.set_ylabel('error')
    ax.legend(loc='upper right')
    save_path = os.path.join('model', '{}_error'.format(model_prefix))
    plt.savefig(save_path, bbox_inches='tight')

if __name__ == '__main__':
    main()
