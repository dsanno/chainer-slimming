import argparse
import json
import numpy as np

import chainer
from chainer import serializers

import net
from net import BatchConv2D


def _parse_args():
    parser = argparse.ArgumentParser('Pruning for network slimming')
    parser.add_argument('src_model', type=str,
                        help='Source model file path')
    parser.add_argument('src_structure', type=str,
                        help='Source structure JSON file path')
    parser.add_argument('dest_model', type=str,
                        help='Destination model file path')
    parser.add_argument('dest_structure', type=str,
                        help='Destination structure JSON file path')
    parser.add_argument('prune_ratio', type=float,
                        help='Prune ratio')
    return parser.parse_args()


def _copy_linear(src, dest, input_mask):
    dest.W.data[...] = src.W.data[:,input_mask]
    if dest.b is not None:
        dest.b.data[...] = src.b.data[...]

def _copy_conv_2d(src, dest, input_mask, output_mask):
    if input_mask is None:
        dest.W.data[...] = src.W.data[output_mask,:,:,:]
    else:
        dest.W.data[...] = src.W.data[output_mask][:,input_mask,:,:]
    if src.b is not None:
        dest.b.data[...] = src.b.data[output_mask]


def _copy_batch_normalization(src, dest, output_mask):
    dest.avg_mean[...] = src.avg_mean[output_mask]
    dest.avg_var[...] = src.avg_var[output_mask]
    dest.N = src.N
    if hasattr(src, 'gamma'):
        dest.gamma.data[...] = src.gamma.data[output_mask]
    if hasattr(src, 'beta'):
        dest.beta.data[...] = src.beta.data[output_mask]


def _copy_layer(src, dest, input_mask, output_mask):
    if isinstance(src, BatchConv2D):
        _copy_conv_2d(src.conv, dest.conv, input_mask, output_mask)
        _copy_batch_normalization(src.bn, dest.bn, output_mask)
    else:
        # FC layer
        _copy_linear(src, dest, input_mask)


def _layer_size(layer):
    size = 0
    if isinstance(layer, BatchConv2D):
        size += np.prod(layer.conv.W.data.shape)
        if layer.conv.b is not None:
            size += np.prod(layer.conv.b.data.shape)
        size += np.prod(layer.bn.avg_mean.shape)
        size += np.prod(layer.bn.avg_var.shape)
        if hasattr(layer.bn, 'gamma'):
            size += np.prod(layer.bn.gamma.data.shape)
        if hasattr(layer.bn, 'beta'):
            size += np.prod(layer.bn.beta.data.shape)
    else:
        # FC layer
        size += np.prod(layer.W.data.shape)
        if layer.b is not None:
            size += np.prod(layer.b.data.shape)
    return size


def _convert(src, dest, mask_dict):
    input_mask = None
    for name in net.VGG_LAYERS:
        output_mask = mask_dict.get(name, None)
        _copy_layer(getattr(src, name), getattr(dest, name), input_mask,
            output_mask)
        input_mask = output_mask


def _find_batch_convs(chain):
    batch_convs = []
    for child in chain.children():
        if isinstance(child, BatchConv2D):
            batch_convs.append(child)
        elif isinstance(child, chainer.Chain):
            batch_convs = batch_convs + _find_batch_convs(child)
    return batch_convs


def _prune(model, prune_ratio):
    mask_dict = {}
    batch_convs = _find_batch_convs(model)
    gammas = []

    for batch_conv in batch_convs:
        gammas.append(batch_conv.bn.gamma.data)
    abs_gammas = abs(np.concatenate(gammas))
    abs_gammas.sort()
    threshold = abs_gammas[int(np.floor(len(abs_gammas) * prune_ratio))]

    for batch_conv in batch_convs:
        gamma = batch_conv.bn.gamma.data
        mask = (abs(gamma) > threshold)
        mask_dict[batch_conv.name] = mask
    return mask_dict


def main():
    args = _parse_args()
    with open(args.src_structure) as f:
        src_output_sizes = json.load(f)
    src_net = net.VGG(src_output_sizes)
    serializers.load_npz(args.src_model, src_net)
    mask_dict = _prune(src_net, args.prune_ratio)
    dest_output_sizes = {k: np.sum(v) for k, v in mask_dict.items()}
    dest_net = net.VGG(dest_output_sizes)
    _convert(src_net, dest_net, mask_dict)
    with open(args.dest_structure, 'w') as f:
        json.dump(dest_output_sizes, f)
    serializers.save_npz(args.dest_model, dest_net)

    # print parameter size
    print('Source model')
    total_size = 0
    for name in net.VGG_LAYERS:
        size = _layer_size(getattr(src_net, name))
        total_size += size
        print('  {}: {}'.format(name, size))
    print('  Total: {}'.format(total_size))
    print('Destination model')
    total_size = 0
    for name in net.VGG_LAYERS:
        size = _layer_size(getattr(dest_net, name))
        total_size += size
        print('  {}: {}'.format(name, size))
    print('  Total: {}'.format(total_size))


if __name__ == '__main__':
    main()
