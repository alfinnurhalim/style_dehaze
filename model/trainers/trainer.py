import random
import numpy as np
import os
import cv2

import torch
import torch.nn as nn
from torchvision.utils import save_image

import model.network.net as net
from model.network.glow import Glow
from model.utils.utils import IterLRScheduler, remove_prefix
from model.layers.activation import calc_mean_std
from model.losses.tv_loss import TVLoss

import logger

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, filename):
    torch.save(state, filename + '.pth.tar')

def get_smooth(I, direction):
    weights = torch.tensor([[0., 0.], [-1., 1.]]).cuda()
    w_x = weights.view(1, 1, 2, 2)
    w_y = weights.t().view(1, 1, 2, 2)
    w = w_x if direction == 'x' else w_y
    return torch.abs(torch.nn.functional.conv2d(I, w, padding=1))

def avg_pool(R, direction):
    return nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(get_smooth(R, direction))

def get_gradients_loss(I, R):
    R_gray = torch.mean(R, dim=1, keepdim=True)
    I_gray = torch.mean(I, dim=1, keepdim=True)
    grad_x = get_smooth(I_gray, 'x')
    grad_y = get_smooth(I_gray, 'y')
    return torch.mean(
        grad_x * torch.exp(-10 * avg_pool(R_gray, 'x')) +
        grad_y * torch.exp(-10 * avg_pool(R_gray, 'y'))
    )

class merge_model(nn.Module):
    def __init__(self, cfg):
        super(merge_model, self).__init__()
        self.glow = Glow(
            in_channel=3,
            n_flow=cfg['n_flow'],
            n_block=cfg['n_block'],
            affine=cfg['affine'],
            conv_lu=not cfg['no_lu']
        )

    def forward(self, content_images, style_code):
        z_c = self.glow(content_images, forward=True)
        stylized = self.glow(z_c, forward=False, style=style_code)
        return stylized


class Trainer:
    def __init__(self, cfg, seed=0):
        set_random_seed(seed)
        self.cfg = cfg
        self.init = True

        # Merge model and optimizer
        self.model = merge_model(cfg).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg['lr'])
        self.lr_scheduler = IterLRScheduler(
            self.optimizer,
            cfg.get('lr_steps', []),
            cfg.get('lr_mults', []),
            last_iter=cfg.get('last_iter', 0)
        )

        # Content encoder (VGG) & Style encoder
        # Load pretrained VGG from torchvision
        vgg = net.vgg
        vgg.load_state_dict(torch.load(cfg['vgg']))
        self.encoder = net.Net(vgg,cfg['keep_ratio']).cuda()

        # Smoothness regularization
        self.tv_loss = TVLoss().cuda()

        # Logging paths
        self.log_path = os.path.join(
            cfg['output'],
            f"{cfg['job_name']}_{int(cfg.get('keep_ratio',1)*100)}_{cfg['n_flow']}_{cfg['n_block']}"
        )
        self.model_log_path = os.path.join(self.log_path, 'model_save')
        self.img_log_path = os.path.join(self.log_path, 'img_save')
        self.img_test_path = os.path.join(self.log_path, 'test_save')
        self.img_att_path = os.path.join(self.log_path, 'att_save')
        for path in [self.model_log_path, self.img_log_path, self.img_test_path, self.img_att_path]:
            os.makedirs(path, exist_ok=True)

    def load_model(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        self.model.load_state_dict(remove_prefix(ckpt['state_dict'], 'module.'))
        self.optimizer.load_state_dict(ckpt['optimizer'])

    def train(self, batch_id, content_imgs, style_imgs, gt_imgs, epoch):
        content = content_imgs.cuda()
        style = style_imgs.cuda()
        gt = gt_imgs.cuda()

        style_code = self.encoder.cat_tensor(style)

        stylized = self.model(content, style_code)
        stylized = torch.clamp(stylized, 0, 1)

        # Smoothness loss
        if self.cfg.get('loss', 'tv') == 'tv':
            loss_smooth = self.tv_loss(stylized)
        else:
            loss_smooth = get_gradients_loss(stylized, style)

        loss_c, loss_s, loss_r = self.encoder(content, style, stylized, gt)

        loss_c = loss_c.mean() * self.cfg.get('content_weight', 1.0)
        loss_s = loss_s.mean() * self.cfg.get('style_weight', 1e-4)
        loss_r = loss_r.mean() * self.cfg.get('recon_weight', 1.0)
        loss_smooth = 

        total_loss = loss_c + loss_s + loss_r + loss_smooth

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        # Logging
        if batch_id % self.cfg.get('log_freq', 100) == 0:
            fname = f"{epoch}_{batch_id}.jpg"
            out = torch.cat([
                content[-1:], style[-1:], stylized[-1:], gt[-1:]
            ], dim=3)
            save_image(out.cpu(), os.path.join(self.img_log_path, fname))
            print('saved at ',os.path.join(self.img_log_path, fname))

        # Checkpointing
        if batch_id % self.cfg.get('save_freq', 500) == 0:
            save_checkpoint(
                {'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()},
                os.path.join(self.model_log_path, f"step_{batch_id}")
            )

        return [
            total_loss.item(),
            loss_c.item(),
            loss_s.item(),
            loss_r.item(),
            loss_smooth.item()
        ]

    def test(self, epoch, batch_id, content_imgs, style_imgs, gt_imgs):
        content = content_imgs.cuda()
        style = style_imgs.cuda()
        gt = gt_imgs.cuda()

        # style_code, mu, logvar = self.encoder.style_encoder(style)
        style_code = self.encoder.cat_tensor(style)
        stylized = self.model(content, style_code)

        stylized = torch.clamp(stylized, 0, 1)

        # Save test outputs

        # Save attention overlays (only if attention enabled)
        if hasattr(self.model.glow, 'blocks'):
            for block_idx, block in enumerate(self.model.glow.blocks):
                if hasattr(block, 'last_attention'):
                    attn_maps = block.last_attention  # [B, 1, H, W]

                    for img_idx in range(attn_maps.shape[0]):
                        attn_overlay =logger.overlay_attention_on_image(
                            attn_maps[img_idx], content[img_idx]
                        )
                        save_name = f"epoch{epoch}_batch{batch_id}_idx{img_idx}_attn_block{block_idx}.jpg"
                        save_path = os.path.join(self.img_att_path, save_name)

                        cv2.imwrite(save_path, cv2.cvtColor(attn_overlay, cv2.COLOR_RGB2BGR))

        for i in range(content.size(0)):
            out = torch.cat([
                content[i:i+1], style[i:i+1], stylized[i:i+1], gt[i:i+1]
            ], dim=3)
            name = f"epoch{epoch}_batch{batch_id}_idx{i}.jpg"
            save_image(out.cpu(), os.path.join(self.img_test_path, name))