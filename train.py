import visdom
import time, sys, os
import argparse
import opencv_transforms.transforms as TF
import dataloader
import functions
import torch
import torch.nn as nn
import torchvision.models
from model import Sketch2Color

# To ignore warning
import warnings
warnings.simplefilter("ignore", UserWarning)

def init_visdom(vis, image_shape):
    B, C, H, W = image_shape
    palette = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='palette'))
    errG2plot = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='errG'))
    loss_l12plot = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='loss_l1'))
    feature2plot = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='feature'))
    tv2plot = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='tv'))

    sketch_img = vis.images(torch.Tensor(B, C, H, W), opts=dict(title='sketch'))
    gt_plt_img = vis.images(torch.Tensor(B, C, H, W), opts=dict(title='gt_plt'))
    gen_plt_img = vis.images(torch.Tensor(B, C, H, W), opts=dict(title='gen_plt'))
    color_img = vis.images(torch.Tensor(B, C, H, W), opts=dict(title='color'))
    generated_img = vis.images(torch.Tensor(B, C, H, W), opts=dict(title='generated'))

    vis_elem = {
        'palette': palette,
        'errG2plot': errG2plot,
        'loss_l1': loss_l12plot,
        'feature': feature2plot,
        'tv': tv2plot,
        'gt_plt_img': gt_plt_img,
        'gen_plt_img': gen_plt_img,
        'sketch_img': sketch_img,
        'color_img': color_img,
        'generated_img': generated_img
    }

    return vis, vis_elem

def init_loss():
    nth_list = []
    palette = []
    errG = []
    loss_l1 = []
    feature = []
    tv = []

    loss_dict = {
        'nth': nth_list,
        'palette': palette,
        'errG2plot': errG,
        'loss_l1': loss_l1,
        'feature': feature,
        'tv': tv,
    }
    return loss_dict

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

def save(sketch2color, epoch, loss_dict):
    global optimizer_G
    print('Saving...', end=' ')
    state = {
        'epoch': epoch,
        'Sketch2Color': sketch2color.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'loss_dict': loss_dict
    }
    if not os.path.isdir('checkpoint'):
        os.makedirs('./checkpoint/', exist_ok=True)
    torch.save(state, './checkpoint/ckpt_{:d}.pth'.format(epoch+1))
    print("Done!")

def load(netG, optG, epoch):
    print('Loading...', end=' ')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_{:d}.pth'.format(epoch))
    netG.load_state_dict(checkpoint['Sketch2Color'], strict=True)
    optG.load_state_dict(checkpoint['optimizer_G']),
    return checkpoint['epoch'], checkpoint['loss_dict']

def get_args_parser():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--train_dir', type=str, required=True, help='directory for train')
    parser.add_argument('--test_sketch_dir', type=str, required=True, help='directory for test sketch')
    parser.add_argument('--test_ref_dir', type=str, required=True, help='directory for test reference')

    # training options
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ncluster', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)

    # lambdas for loss function
    parser.add_argument('--lambda1', type=float, default=10)
    parser.add_argument('--lambda2', type=float, default=1e-4)
    parser.add_argument('--lambda3', type=float, default=1e-2)

    # betas for optimizer
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # transformer
    parser.add_argument('--transformer_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--feedforward_dim', type=int, default=512)
    parser.add_argument('--num_enc_layers', type=int, default=6)
    parser.add_argument('--num_dec_layers', type=int, default=6)

    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Visdom display initialization
    vis = visdom.Visdom()
    vis.close(env="main")
    vis, vis_elem = init_visdom(vis, (args.batch_size, 3, args.img_size, args.img_size))
    loss_dict = init_loss()

    # Training data
    print('Loading Training data...', end=' ')
    train_transforms = TF.Compose([
        # TF.RandomResizedCrop(256),
        TF.Resize(256),
        TF.RandomHorizontalFlip(),
        ])

    train_image_folder = dataloader.FlatFolderDataset(args.train_dir, train_transforms, args, palette_info=True)
    train_loader = torch.utils.data.DataLoader(train_image_folder, batch_size=args.batch_size, shuffle=True)
    print("Done!")
    print("Training data size : {}".format(len(train_image_folder)))
    train_batch = next(iter(train_loader))

    # Visdom for test visualization ##############################################
    test_transform = TF.Compose([ # ToTensor and Normalize are in dataloader
        TF.Resize(256)
        ])

    test_image_folder = dataloader.FlatFolderDataset(args.test_sketch_dir, test_transform, args, palette_info=False)
    test_loader = torch.utils.data.DataLoader(test_image_folder, batch_size=args.batch_size, shuffle=True)
    print("Done!")
    print("Test data size : {}".format(len(test_image_folder)))
    test_batch = next(iter(test_loader))

    refer_image_folder = dataloader.FlatFolderDataset(args.test_ref_dir, test_transform, args, palette_info=False)
    refer_loader = torch.utils.data.DataLoader(refer_image_folder, batch_size=args.batch_size, shuffle=True)
    refer_batch = next(iter(refer_loader))
    print("Done!")
    print("Reference data size : {}".format(len(refer_image_folder)))
    ##############################################################################

    model = Sketch2Color(args).to(device)
    netEx = torchvision.models.vgg16(True).features[0:4].to(device)

    torch.backends.cudnn.benchmark = True

    # Loss functions
    criterion_L1 = torch.nn.L1Loss()  # L1 Loss
    criterion_L2 = torch.nn.MSELoss()  # L2 Loss
    criterion_TV = TVLoss()  # Total variance Loss

    optimizer_G = torch.optim.Adam(model.parameters(),lr=args.lr, betas=(args.beta1, args.beta2))

    model.train()

    print("Starting Training Loop...")
    # For each epoch
    current_epoch = 0
    last_epoch = args.max_epoch + current_epoch - 1

    m = torch.nn.Upsample(scale_factor=16)

    for epoch in range(current_epoch, args.max_epoch + current_epoch):
        current_epoch += 1

        start_time = time.time()
        total_time = 0

        print('Epoch [{0}/{1}]'.format(epoch, last_epoch))
        for i, data in enumerate(train_loader, 0):

            edge = data[0].to(device) # (B, 1, 256, 256)
            color = data[1].to(device) # (B, 3, 256, 256)
            gt_palette = data[2].to(device) # torch.Size([8, 12, 1, 1]) (B, ncluster*channel, size, size)
            b_size = edge.size(0)

            fake, fake_palette = model(edge, color)

            fake_feature = netEx(fake)
            real_feature = netEx(color)

            optimizer_G.zero_grad()

            loss_P = criterion_L2(gt_palette, fake_palette)
            loss_L1 = criterion_L1(fake, color)
            loss_TV = criterion_TV(fake)
            loss_Feature = criterion_L2(fake_feature, real_feature)

            # Total loss
            loss_G = args.lambda1 * loss_L1 + args.lambda2 * loss_TV + args.lambda3 * loss_Feature + loss_P
            loss_G.backward()

            optimizer_G.step()
            ###################################

            if i % 10 == 0:
                # Time Info.
                end_time = time.time()
                taken_time = end_time - start_time
                total_time += taken_time
                average_time = total_time / (i + 1)

                # Output training stats
                print('\r[%d/%d] Loss_P: %.2f / Loss_G: %.2f / Loss_L1: %.2f /' \
                      ' Loss_TV: %.2f / Loss_Feature: %.2f / Time : %.2f (%.2f)'
                      % (i + 1, len(train_loader), loss_P.item(), loss_G.item(), loss_L1.item(), loss_TV, loss_Feature, \
                         taken_time, average_time), end='     ')
                start_time = end_time

            # Log Results
            loss_dict['palette'].append(loss_P.cpu().item())
            loss_dict['errG2plot'].append(loss_G.cpu().item())
            loss_dict['loss_l1'].append(args.lambda1 * loss_L1.item())
            loss_dict['feature'].append(args.lambda3 * loss_Feature)
            loss_dict['tv'].append(args.lambda2 * loss_TV)

            nth = epoch * len(train_loader) + i
            loss_dict['nth'].append(nth)


            if i % 50 == 0 or i + 1 == len(train_loader):
                vis.line(Y=torch.Tensor(loss_dict['palette']), X=torch.Tensor(loss_dict['nth']),
                         win=vis_elem['palette'], opts=dict(title='palette'))
                vis.line(Y=torch.Tensor(loss_dict['errG2plot']), X=torch.Tensor(loss_dict['nth']),
                         win=vis_elem['errG2plot'], opts=dict(title='errG'))
                vis.line(Y=torch.Tensor(loss_dict['loss_l1']), X=torch.Tensor(loss_dict['nth']), win=vis_elem['loss_l1'],
                         opts=dict(title='loss_l1'))
                vis.line(Y=torch.Tensor(loss_dict['feature']), X=torch.Tensor(loss_dict['nth']), win=vis_elem['feature'],
                         opts=dict(title='feature'))
                vis.line(Y=torch.Tensor(loss_dict['tv']), X=torch.Tensor(loss_dict['nth']), win=vis_elem['tv'],
                         opts=dict(title='tv'))

                vis.images(functions.denorm(edge.detach()), win=vis_elem['sketch_img'], opts=dict(title='sketch_img'))
                vis.images(functions.denorm(color.detach()), win=vis_elem['color_img'], opts=dict(title='color_img'))
                vis.images(functions.denorm(fake.detach()), win=vis_elem['generated_img'],
                               opts=dict(title='generated_img'))

                gt_palette = m(gt_palette)
                gt_palette = gt_palette.reshape(args.batch_size, args.ncluster, 3, 16, 16)
                gt_palette = gt_palette.reshape(args.batch_size*args.ncluster, 3, 16, 16) #[8, 48, 1, 1]
                fake_palette = m(fake_palette)
                fake_palette = fake_palette.reshape(args.batch_size, args.ncluster, 3, 16, 16)
                fake_palette = fake_palette.reshape(args.batch_size * args.ncluster, 3, 16, 16)

                vis.images(functions.denorm(gt_palette.detach()), win=vis_elem['gt_plt_img'],
                                opts=dict(title='gt_plt_img')) # torch.Size([8, 12, 1, 1])
                vis.images(functions.denorm(fake_palette.detach()), win=vis_elem['gen_plt_img'],
                               opts=dict(title='gen_plt_img'))

                # Visualization for test images ##############################
                with torch.no_grad():
                    edge_test = test_batch[0].to(device)
                    real_test = test_batch[1].to(device)
                    reference_test = refer_batch[1].to(device)

                    out_test, gen_palette_test = model(edge_test, reference_test)

                    gen_palette_test = m(gen_palette_test)
                    gen_palette_test = gen_palette_test.reshape(args.batch_size, args.ncluster, 3, 16, 16)
                    gen_palette_test = gen_palette_test.reshape(args.batch_size * args.ncluster, 3, 16, 16)

                    vis.images(functions.denorm(edge_test.detach()), win='test_sketch', opts=dict(title='test_sketch'))
                    vis.images(functions.denorm(reference_test.detach()), win='test_gt', opts=dict(title='test_gt'))
                    vis.images(functions.denorm(out_test.detach()), win='test_generated',
                               opts=dict(title='test_generated'))
                    vis.images(functions.denorm(gen_palette_test.detach()), win='test_gen_palette',
                               opts=dict(title='test_gen_palette'))
                ###############################################################

        print("Current epoch : {}".format(current_epoch))
        save(model, current_epoch, loss_dict)

    print('Training Done')