"""
Evaluate supervised regression on CelebA-HQ 128x128
"""
# imports
import numpy as np
import torch
from datasets.celeba_dataset import evaluate_lin_reg_on_mafl_topk
from models import KeyPointVAE

if __name__ == '__main__':
    image_size = 128
    imwidth = 160
    crop = 16
    ch = 3
    # enc_channels = [64, 128, 256, 512]
    enc_channels = [32, 64, 128, 256]
    # prior_channels = (16, 16, 32)
    prior_channels = (16, 32, 64)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    use_logsoftmax = False
    # pad_mode = 'zeros'
    pad_mode = 'reflect'
    sigma = 0.1  # default sigma for the gaussian maps
    n_kp = 1  # num kp per patch
    n_kp_enc = 30  # total kp to output from the encoder / filter from prior
    n_kp_prior = 64
    mask_threshold = 0.2  # mask threshold for the features from the encoder
    patch_size = 8  # 8 for playground, 16 for celeb
    learned_feature_dim = 10  # additional features than x,y for each kp
    # kp_range = (0, 1)
    kp_range = (-1, 1)
    # dec_bone = "gauss_pointnetpp"
    dec_bone = "gauss_pointnetpp_feat"
    topk = 10
    # kp_activation = "none"
    # kp_activation = "sigmoid"
    kp_activation = "tanh"
    use_object_enc = True  # separate object encoder
    use_object_dec = False  # separate object decoder
    anchor_s = 0.125
    learn_order = False
    kl_balance = 0.001
    dropout = 0.0
    root = '/mnt/data/tal/celeba'
    path_to_model_ckpt = './best30_050122_131101_celeba_var_particles_gauss_pointnetpp_feat/' \
                         'saves/celeba_var_particles_gauss_pointnetpp_feat_best.pth'

    model = KeyPointVAE(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                        image_size=image_size, n_kp=n_kp, learned_feature_dim=learned_feature_dim,
                        use_logsoftmax=use_logsoftmax, pad_mode=pad_mode, sigma=sigma,
                        dropout=dropout, dec_bone=dec_bone, patch_size=patch_size, n_kp_enc=n_kp_enc,
                        n_kp_prior=n_kp_prior, kp_range=kp_range, kp_activation=kp_activation,
                        mask_threshold=mask_threshold, use_object_enc=use_object_enc,
                        use_object_dec=use_object_dec, anchor_s=anchor_s, learn_order=learn_order).to(device)
    model.load_state_dict(
        torch.load(path_to_model_ckpt, map_location=device))
    print("loaded model from checkpoint")
    print('evaluating linear regression')
    result = evaluate_lin_reg_on_mafl_topk(model, root=root, batch_size=100, device=device, topk=topk)
    print(result)