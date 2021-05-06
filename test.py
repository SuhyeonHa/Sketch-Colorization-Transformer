import math

from PIL import Image
import requests
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import opencv_transforms.transforms as TF

import os
from model_res_tf_plt import Sketch2Color
import dataloader_flat_test
import torchvision.utils as vutils
import numpy as np
import IPython

torch.set_grad_enabled(False)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device=='cuda':
    print("The gpu to be used : {}".format(torch.cuda.get_device_name(0)))
else:
    print("No gpu detected")

# configs ######################################
img_size = 256
transformer_dim = 128
# learning rate
lr = 2e-4
# Loss functions
criterion_GAN = torch.nn.MSELoss() # LSGAN
criterion_L1 = torch.nn.L1Loss() # L1 Loss
criterion_L2 = torch.nn.MSELoss() # L2 Loss
# criterion_TV = TVLoss() # Total variance Loss
# Lambda
lambda1 = 100
lambda2 = 1e-4
lambda3 = 1e-2
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
beta2 = 0.999

batch_size = 1
ncluster = 16
# configs ######################################
test_transform = TF.Compose([ # ToTensor and Normalize in dataloader
    TF.Resize(256)
    ])

model = Sketch2Color(img_size=img_size, trs_dim=transformer_dim, ncluster=ncluster)
optimizer_G = torch.optim.Adam(model.parameters(),lr=lr, betas=(beta1, beta2))

m = torch.nn.Upsample(scale_factor=16)

def load(model, optimizer_G):
    # global current_epoch, best_losses, loss_list_D, loss_list_G, optimizer_G, optimizer_D
    print('Loading...', end=' ')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('./checkpoint/edge2color/ckpt.pth')
    checkpoint = torch.load('./checkpoint\edge2color\save/e40_n16_tf_plt.pth', map_location=device)
    #current_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['Sketch2Color'], strict=True)
    # netD.load_state_dict(checkpoint['netD'], strict=True)
    # loss_list_D = checkpoint['loss_list_D'],
    # loss_list_G = checkpoint['loss_list_G'],
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    # optimizer_D.load_state_dict(checkpoint['optimizer_D']),
    return model, optimizer_G

model, optimizer_G = load(model, optimizer_G)
model.eval()

test_imagefolder = dataloader_flat_test.FlatFolderDataset('D:\Dataset\sketch-pair/val', test_transform)
test_loader = torch.utils.data.DataLoader(test_imagefolder, batch_size=batch_size, shuffle=True)
print("Done!")
print("Test data size : {}".format(len(test_imagefolder)))
test_batch = next(iter(test_loader))
# temp_batch_iter = iter(train_loader)

# Reference
print('Loading Reference data...', end=' ')
# refer_transforms = TF.Compose([
#     TF.Resize(512),
#     ])
refer_imagefolder = dataloader_flat_test.FlatFolderDataset('D:\Dataset\sketch-pair/val', test_transform)
refer_loader = torch.utils.data.DataLoader(refer_imagefolder, batch_size=batch_size, shuffle=True)
refer_batch = next(iter(refer_loader))
print("Done!")
print("Reference data size : {}".format(len(refer_imagefolder)))

# use lists to store the outputs via up-values
# conv_features, enc_attn_weights, dec_attn_weights = [], [], []
#
# hooks = [
#     model.backbone[-2].register_forward_hook(
#         lambda self, input, output: conv_features.append(output)
#     ),
#     model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
#         lambda self, input, output: enc_attn_weights.append(output[1])
#     ),
#     model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
#         lambda self, input, output: dec_attn_weights.append(output[1])
#     ),
# ]

e_conv_features, c_conv_features, enc_attn_weights, dec_attn_weights = [], [], [], []

hooks = [
    model.e_encoder.register_forward_hook(
        lambda self, input, output: e_conv_features.append(output[0])
    ),
    model.c_encoder.register_forward_hook(
        lambda self, input, output: c_conv_features.append(output[0])
    ),
    model.transformer.e_encoder.layers[-1].self_attn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output)
    ),
    model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
        lambda self, input, output: dec_attn_weights.append(output)
    ),
]

with torch.no_grad():
    edge = test_batch[0]
    real = test_batch[1]
    reference = refer_batch[1]

    out, out_palette = model(edge, reference)

    gt_palette = m(out_palette)
    gt_palette = gt_palette.reshape(batch_size, ncluster, 3, 16, 16)
    gt_palette = gt_palette.reshape(batch_size * ncluster, 3, 16, 16)  # [8, 48, 1, 1]

    plt.figure(figsize=(16, 16))
    result = torch.cat([torch.cat((edge, edge, edge), dim=1), reference, out, real], dim=-1)
    plt.imshow(np.transpose(vutils.make_grid(result, nrow=1, padding=5, normalize=True).cpu(), (1, 2, 0)))
    plt.axis("off")
    plt.title("Sketch / Reference / Generated / GT", fontsize=10)
    plt.show()

    for hook in hooks:
        hook.remove()

    e_conv_features = e_conv_features[0][0]
    c_conv_features = c_conv_features[0][0]
    # enc_attn_weights = enc_attn_weights[0][0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]

    # """Now let's visualize them"""
    #
    # # get the feature map shape
    # h, w = c_conv_features['0'].tensors.shape[-2:]
    #
    # fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    # colors = COLORS * 100
    # for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    #     ax = ax_i[0]
    #     ax.imshow(dec_attn_weights[0, idx].view(h, w))
    #     ax.axis('off')
    #     ax.set_title(f'query id: {idx.item()}')
    #     ax = ax_i[1]
    #     ax.imshow(im)
    #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                fill=False, color='blue', linewidth=3))
    #     ax.axis('off')
    #     ax.set_title(CLASSES[probas[idx].argmax()])
    # fig.tight_layout()

    # output of the CNN
    f_map_e = e_conv_features # torch.Size([1, 992, 16, 16])
    f_map_c = c_conv_features # torch.Size([1, 992, 16, 16])
    print("Encoder attention:      ", enc_attn_weights[0].shape)
    print("Feature map:            ", f_map_e.shape)
    print("Feature map:            ", f_map_c.shape)

    # get the HxW shape of the feature maps of the CNN
    # shape = f_map_e.shape[-2:]
    # and reshape the self-attention to a more interpretable shape
    sattn = enc_attn_weights[0].reshape(shape + shape)
    print("Reshaped self-attention:", sattn.shape)

    # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
    fact = 16

    # let's select 4 reference points for visualization
    idxs = [(64, 64), (64, 128), (128, 128), (192, 128), ]

    # here we create the canvas
    fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
    # and we add one plot per reference point
    gs = fig.add_gridspec(2, 4)
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[0, -1]),
        fig.add_subplot(gs[1, -1]),
    ]

    # for each one of the reference points, let's plot the self-attention
    # for that point
    for idx_o, ax in zip(idxs, axs):
        idx = (idx_o[0] // fact, idx_o[1] // fact)
        ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'self-attention{idx_o}')

    # and now let's add the central image, with the reference points as red circles
    fcenter_ax = fig.add_subplot(gs[:, 1:-1])
    fcenter_ax.imshow(edge.squeeze())
    for (y, x) in idxs:
        # scale = edge.height / edge.shape[-2]
        # x = ((x // fact) + 0.5) * fact
        # y = ((y // fact) + 0.5) * fact
        # fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
        fcenter_ax.add_patch(plt.Circle((x, y), fact // 2, color='r'))
        fcenter_ax.axis('off')
    plt.show()