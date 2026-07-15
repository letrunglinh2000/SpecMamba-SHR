import os.path
import math
import argparse
import time
import random
import numpy as np
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

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = False #parser.parse_args().dist

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
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    init_iter_VAU, init_path_VAU = option.find_last_checkpoint(opt['path']['models'], net_type='VAU')
    # init_path_G = './weight/SHIQ.pth'
    # init_iter_E = init_path_G
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    opt['path']['pretrained_netVAU'] = init_path_VAU
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG, init_iter_VAU)

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


        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False)


    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    print('Number of pamrams: ',count_parameters(model.netG))

    print('Number of pamrams VAUnet: ',count_parameters(model.VAUNet))
    #if opt['rank'] == 0:
    #    logger.info(model.info_network())
    #   logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''


    
    avg_psnr = 0.0
    idx = 0
    avg_ssim = 0.0
    num_img = 0
    count = 0
    
    start = time.perf_counter()
    
    
    for test_data in test_loader:
        idx += 1
        

        num_img += 1
        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

        img_dir = os.path.join('Output','SD2v2','Removal' )
        util.mkdir(img_dir)
        img_dir = os.path.join('Output','SD2v2','Mask' )
        util.mkdir(img_dir)
        #print(test_data)
        #window_size = 8
        # _, _, h_old, w_old = test_data['L'].size()
        # h_pad = (h_old // window_size + 1) * window_size - h_old
        # w_pad = (w_old // window_size + 1) * window_size - w_old
        # test_data['L'] = torch.cat([test_data['L'], torch.flip(test_data['L'], [2])], 2)[:, :, :h_old + h_pad, :]
        # test_data['L'] = torch.cat([test_data['L'], torch.flip(test_data['L'], [3])], 3)[:, :, :, :w_old + w_pad]
        
        model.feed_data(test_data)
        model.test()

        visuals = model.current_visuals()
        #E_img = util.tensor2uint(visuals['E'])
        #H_img = util.tensor2uint(visuals['H'])


        if True:
            E_img = util.quatertensor2uint(visuals['E'])
            H_img = util.quatertensor2uint(visuals['H'])
            M_img = util.tensor2uint(visuals['M'])
            #MGT_img = util.tensor2uint(visuals['MGT'])
            #print(M_img.shape)
            #print(MGT_img.shape)

            save_img_path = os.path.join('Output','SD2v2','Removal', '{:s}.png'.format(img_name ))
            save_mask_path = os.path.join('Output','SD2v2','Mask', '{:s}.png'.format(img_name))
            #save_maskGT_path = os.path.join(img_dir, 'MaskGT_{:s}_{:d}.png'.format(img_name, current_step))
            
            
            util.imsave(E_img, save_img_path)
            util.imsave(M_img, save_mask_path)
            #util.imsave(MGT_img, save_maskGT_path)
            
            current_psnr = util.calculate_psnr(E_img, H_img, border=border,test_y_channel=False)
            current_ssim = util.calculate_ssim(E_img, H_img, border=0, scale=1)
            print('{:->4d}--> {:>10s} | {:<4.2f}dB/{:<4.4f}dB'.format(idx, img_name, current_psnr,current_ssim))
            avg_psnr += current_psnr
            avg_ssim += current_ssim

        # -----------------------
        # save estimated image E
        # -----------------------
        
    end = time.perf_counter()

    elapsed = end - start
    print(f"Elapsed time: {elapsed:.2f} seconds")

    avg_psnr = avg_psnr / num_img
    avg_ssim = avg_ssim / num_img
    print(' Average PSNR/SSIM in cutting way: {:<.2f}dB/{:<.4f}dB\n'.format(avg_psnr, avg_ssim))


if __name__ == '__main__':
    main()