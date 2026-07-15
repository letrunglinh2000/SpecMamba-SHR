import os.path
import math
import argparse
import time
import random
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(json_path='options/train_qformer.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    parser.add_argument('--inp_path', default='trainsets/SHIQ/Inp/')
    parser.add_argument('--gt_path', default='trainsets/SHIQ/Out/')
    parser.add_argument('--weight', default='./finetune/37.26/185000_G.pth')
    parser.add_argument('--weightSpecMamba', default='.//finetune/37.26/185000_VAU.pth')
    parser.add_argument('--weight_t_embed', default=None,
                        help='Path to t_embed.pth (required when loading a diffusion-trained checkpoint)')
    parser.add_argument('--infer_steps', type=int, default=None,
                        help='Override number of DDIM steps (default: use config)')


    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist
    opt['path']['pretrained_netG'] = parser.parse_args().weight
    opt['path']['pretrained_netSpecMamba'] = parser.parse_args().weightSpecMamba
    if parser.parse_args().weight_t_embed is not None:
        opt['path']['pretrained_t_embed'] = parser.parse_args().weight_t_embed
    if parser.parse_args().infer_steps is not None:
        opt['train'].setdefault('diffusion', {})['infer_steps'] = parser.parse_args().infer_steps

    opt['datasets']['test']['dataroot_H']=parser.parse_args().gt_path#'trainsets/SHIQ/Out/'#train_sets/Spec/datasets/spec2diff/testA
    opt['datasets']['test']['dataroot_L'] = parser.parse_args().inp_path#'trainsets/SHIQ/Inp/'
    opt['datasets']['test']['dataroot_M'] = None  # no GT mask at test time
    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))
    #print(opt['path'])
    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-


    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------

    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        #logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------

    for phase, dataset_opt in opt['datasets'].items():

        if phase == 'train':
            continue

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_test()
    #print('Number of parameters: {}'.format(model.info_params()))
    #if opt['rank'] == 0:
    #    logger.info(model.info_network())
    #   logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''


    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    
    pbar = tqdm(
            test_loader,
            total=len(test_loader),
            desc="Testing",
            unit="img",
            dynamic_ncols=True,
            leave=True,
        )
    
    # for test_data in test_loader:
    for idx, test_data in enumerate(pbar, start=1):
        # idx += 1
        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

        img_dir = os.path.join(opt['path']['images'], img_name)
        util.mkdir(img_dir)
        # print(test_data)
        #print(test_data['L'].shape)
        # _, _, h_old, w_old = test_data.size()
        # h_pad = (h_old // window_size + 1) * window_size - h_old
        # w_pad = (w_old // window_size + 1) * window_size - w_old
        # test_data = torch.cat([test_data, torch.flip(test_data, [2])], 2)[:, :, :h_old + h_pad, :]
        # test_data = torch.cat([test_data, torch.flip(test_data, [3])], 3)[:, :, :, :w_old + w_pad]

        # _, _, h_old, w_old = test_data['L'].size()
        # window_size = 8
        # h_pad = (h_old // window_size + 1) * window_size - h_old
        # w_pad = (w_old // window_size + 1) * window_size - w_old
        # test_data['L'] = torch.cat([test_data['L'], torch.flip(test_data['L'], [2])], 2)[:, :, :h_old + h_pad, :]
        # test_data['L'] = torch.cat([test_data['L'], torch.flip(test_data['L'], [3])], 3)[:, :, :, :w_old + w_pad]

        model.feed_data(test_data)
        model.test()

        visuals = model.current_visuals()

        # visuals['E'] and ['H'] are now 3-channel RGB (quaternion channel stripped in model)
        E_img = util.tensor2uint(visuals['E'])
        H_img = util.tensor2uint(visuals['H'])
        M_img = util.tensor2uint(visuals['M'])

        save_img_path = os.path.join('Output', opt['datasets']['test']['dataroot_H'].split('/')[2],'Removal',
                                    '{:s}.png'.format(img_name))
        
        save_img_mask = os.path.join('Output', opt['datasets']['test']['dataroot_H'].split('/')[2],'Mask',
                                    '{:s}.png'.format(img_name))
        directory = os.path.dirname(save_img_path)
        if directory != "" and not os.path.exists(directory):
            os.makedirs(directory)

        directory = os.path.dirname(save_img_mask)
        if directory != "" and not os.path.exists(directory):
            os.makedirs(directory)
        util.imsave(E_img, save_img_path)

        util.imsave(M_img, save_img_mask)

        current_psnr = util.calculate_psnr(E_img, H_img, border=border, test_y_channel=False)
        current_ssim = util.calculate_ssim(E_img, H_img, border=0, scale=1)
        # print('{:->4d}--> {:>10s} | {:<4.2f}dB/{:<4.4f}dB'.format(idx, img_name, current_psnr,current_ssim))
        avg_psnr += current_psnr
        avg_ssim += current_ssim
        
        running_psnr = avg_psnr / idx
        running_ssim = avg_ssim / idx
        pbar.set_postfix({
            "img": img_name,
            "PSNR": f"{current_psnr:.2f}",
            "SSIM": f"{current_ssim:.4f}",
            "Avg_PSNR": f"{running_psnr:.2f}",
            "Avg_SSIM": f"{running_ssim:.4f}",
        })

        # -----------------------
        # save estimated image E
        # -----------------------

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    print(' Average PSNR/SSIM in cutting way: {:<.2f}dB/{:<.4f}dB\n'.format(avg_psnr, avg_ssim))



if __name__ == '__main__':
    main()

