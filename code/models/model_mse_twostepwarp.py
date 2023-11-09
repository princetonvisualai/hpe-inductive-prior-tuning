import sys
sys.path.append("..")
sys.path.append("../..")

import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision.models as models
from networks_baseline import ParameterRegressor, Reconstructor
from utils.losses import compute_anchor_loss, compute_boundary_loss, _l1_loss, _l2_loss
from utils.helper import draw_template, load_anchor_points
from utils.transforms import transform_template, transform_anchor_points
import configs.variables as variables

TEMPLATES_DIR = variables.TEMPLATES_DIR
PRETRAINED_VGG_PATH = variables.PRETRAINED_VGG_PATH

class Model:
    def __init__(self, cfg, device):

        self.device = device
        self.template = draw_template(TEMPLATES_DIR + cfg['template_path'], size=cfg['img_size'], batch_size=cfg['batch_size'],
                                      device=device)
        self.core, self.single, self.double = load_anchor_points(TEMPLATES_DIR + cfg['anchor_pts_path'], device, cfg['batch_size'])
        self.regressor = ParameterRegressor(num_features=cfg['regressor_nf'], num_parts=20).to(device)
        self.translator = Reconstructor(num_features=cfg['translator_nf'], num_parts=cfg['num_parts']).to(device)
        
        self.optim = Adam(list(self.regressor.parameters()) + list(self.translator.parameters()),
                          lr=cfg['learning_rate'])
        
        self.vgg = nn.Sequential()
        # since compute node has no internet, use pre-downloaded vgg model
        vgg = models.vgg19(pretrained=False)
        vgg.load_state_dict(torch.load(PRETRAINED_VGG_PATH))
        vgg = vgg.features.eval().to(device)
        #vgg = models.vgg19(pretrained=True).features.eval().to(device)
        
        depth = 14
        for i in range(depth):
            self.vgg.add_module(str(i), vgg[i])

        self.I = torch.eye(3)[0:2].view(1, 1, 2, 3).repeat(cfg['batch_size'], cfg['num_parts'], 1, 1).to(device)
        self.aug = torch.Tensor([0, 0, 1]).view(1, 1, 1, 3).repeat(cfg['batch_size'], cfg['num_parts'], 1, 1).to(device)
        self.lambda1 = cfg['anchor_loss_weight']
        self.lambda2 = cfg['boundary_loss_weight']

    def train_step(self, frame1, frame2, return_imgs=False, predict=False):
        
        # Get batch_size, number of joints and image dimension
        batch_size = frame1.shape[0]
        num_parts = self.template.shape[1]
        img_size = frame1.shape[2]
        
        # Estimate the regressor parameters
        estimated_params = self.regressor(frame1) # shape: batch_size, 20, 2, 3

        # Split estimated params into two step warp
        estimated_params_1 = torch.zeros(batch_size, num_parts, 2, 3).to(self.device)
        estimated_params_2 = torch.zeros(batch_size, num_parts, 2, 3).to(self.device)

        estimated_params_1[:, 0:9] = estimated_params[:, 0:9] # core, hips, thighs, shins, shoulders (9)
        estimated_params_1[:, [9,11,15]] = estimated_params[:, 9].unsqueeze(1) # left arm, forearm, hand (3)
        estimated_params_1[:, [10,12,16]] = estimated_params[:, 10].unsqueeze(1) # right arm, forearm, hand (3)
        estimated_params_1[:, 13:15] = estimated_params[:, 11:13] # feet 2
        estimated_params_1[:, 17] = estimated_params[:, 13] # head 1

        estimated_params_2[:, 9] = estimated_params[:, 14] # left arm
        estimated_params_2[:, 11] = estimated_params[:, 15] # left forearm
        estimated_params_2[:, 15] = estimated_params[:, 16] # left hand
        estimated_params_2[:, 10] = estimated_params[:, 17] # right arm
        estimated_params_2[:, 12] = estimated_params[:, 18] # right forearm
        estimated_params_2[:, 16] = estimated_params[:, 19] # right hand

        estimated_params_1 = self.I + estimated_params_1
        estimated_params_2 = self.I + estimated_params_2

        # Transform template according to estimated parameters
        # (batch, num_parts) --> (batch*num_parts)

        # WARP STEP 1
        batched_template = self.template.view(-1, img_size, img_size).unsqueeze(1)
        batched_params_1 = estimated_params_1.view(-1, 2, 3)
        transformed_template_1 = transform_template(batched_template, batched_params_1) # estimated warped template       
        # Transform anchor points according to estimated parameters
        A_1 = torch.cat([estimated_params_1, self.aug], dim=-2)
        transformed_anchors_1 = transform_anchor_points(A_1, self.core, self.double, self.single) # estimated anchor points

        # WARP STEP 2
        batched_params_2 = estimated_params_2.view(-1, 2, 3)
        transformed_template_2 = transform_template(transformed_template_1, batched_params_2) # estimated warped template
        # Transform anchor points according to estimated parameters
        A_2 = torch.cat([estimated_params_2, self.aug], dim=-2)
        transformed_anchors_2 = transform_anchor_points(A_2, transformed_anchors_1[0], transformed_anchors_1[1], transformed_anchors_1[2]) # estimated anchor points

        # (batch*num_parts) --> (batch, num_parts)
        transformed_template_1 = transformed_template_1.view(batch_size, num_parts, img_size, img_size)
        transformed_template_2 = transformed_template_2.view(batch_size, num_parts, img_size, img_size)

        # Reconstruct the background of the original frame (frame1), based on frame2 and estimated template
        reconstructed = self.translator(frame2, transformed_template_2)

        transformed_template_1 = transformed_template_1.unsqueeze(2).repeat(1, 1, 3, 1, 1).sum(1)
        transformed_template_2 = transformed_template_2.unsqueeze(2).repeat(1, 1, 3, 1, 1).sum(1)
        
        anchor_loss = compute_anchor_loss(*transformed_anchors_1, size=img_size) + compute_anchor_loss(*transformed_anchors_2, size=img_size)
        boundary_loss = compute_boundary_loss(*transformed_anchors_1, img_size=img_size) + compute_boundary_loss(*transformed_anchors_2, img_size=img_size)
        recon_perceptual_loss = _l1_loss(self.vgg(frame1), self.vgg(reconstructed))
        recon_mse_loss = _l2_loss(frame1, reconstructed)

        # overall loss, gradient and weight update
        loss = recon_perceptual_loss + recon_mse_loss + self.lambda1 * anchor_loss + self.lambda2 * boundary_loss
        
        # if training
        if not predict:
            # reset gradients
            self.regressor.zero_grad()
            self.translator.zero_grad()
            loss.backward()
            self.optim.step()

        d = {'anchor_loss': anchor_loss, 'boundary_loss': boundary_loss, 'recon_perceptual_loss': recon_perceptual_loss,
             'recon_mse_loss': recon_mse_loss}
        if return_imgs:
            d['reconstructed'] = reconstructed
            d['frame1'] = frame1
            d['frame2'] = frame2
            d['transformed_template'] = transformed_template_2.detach()
            d['transformed_template1'] = transformed_template_1.detach()
            
        if predict:
            d['transformed_anchors'] = transformed_anchors_2
        
        return d
