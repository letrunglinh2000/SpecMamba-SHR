import os.path
import argparse
import random
import numpy as np
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
    parser.add_argument('--dist', action='store_true',
                        help='Force distributed mode (auto-enabled if torchrun env is detected)')
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    # Auto-detect distributed mode: any of LOCAL_RANK/RANK/WORLD_SIZE means torchrun.
    torchrun_env = any(k in os.environ for k in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE'))
    opt['dist'] = bool(args.dist or torchrun_env)
    # Single-pass training hits all params, so DDP doesn't need this expensive scan.
    opt['find_unused_parameters'] = False

    # ----------------------------------------
    # Speed knobs (cudnn benchmark + TF32 matmul/cudnn)
    # ----------------------------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        # Eval on rank-0 can take 15+ min; raise the NCCL heartbeat watchdog
        # timeout so it doesn't fire while the other ranks wait at the barrier.
        import os as _os
        _os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")
        _os.environ.setdefault("NCCL_TIMEOUT", "7200")
        init_dist('pytorch')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    opt['rank'], opt['world_size'] = get_dist_info()

    print(
            f"rank={opt['rank']}, "
            f"world_size={opt['world_size']}, "
            f"local_rank={os.environ.get('LOCAL_RANK')}, "
            f"cuda_device={torch.cuda.current_device()}"
        )
    
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))
    #print(opt['path'])
    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    init_iter_SpecMamba, init_path_SpecMamba = option.find_last_checkpoint(opt['path']['models'], net_type='SpecMamba')
    init_iter_t_embed, init_path_t_embed = option.find_last_checkpoint(opt['path']['models'], net_type='t_embed')
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    init_iter_optimizerSpecMamba, init_path_optimizerSpecMamba = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerSpecMamba')

    current_step = 0  # Ensure current_step is always initialized

    if init_iter_G != 0:
        opt['path']['pretrained_netG'] = init_path_G
        opt['path']['pretrained_netE'] = init_path_E
        opt['path']['pretrained_netSpecMamba'] = init_path_SpecMamba
        if init_iter_t_embed != 0:
            opt['path']['pretrained_t_embed'] = init_path_t_embed
        opt['path']['pretrained_optimizerG'] = init_path_optimizerG
        opt['path']['pretrained_optimizerSpecMamba'] = init_path_optimizerSpecMamba
        current_step = max(
            init_iter_G, init_iter_E, init_iter_optimizerG,
            init_iter_SpecMamba, init_iter_optimizerSpecMamba, init_iter_t_embed,
        )

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
            train_set = define_Dataset(dataset_opt)
            if opt['dist']:
                train_sampler = DistributedSampler(
                train_set,
                num_replicas=opt['world_size'],
                rank=opt['rank'],
                shuffle=True,
                drop_last=True,
                )
                train_shuffle = False
            else:
                train_sampler = None
                train_shuffle = dataset_opt['dataloader_shuffle']

            num_workers = int(dataset_opt['dataloader_num_workers'])
            train_loader = DataLoader(
                train_set,
                batch_size=dataset_opt['dataloader_batch_size'],  # per GPU batch size
                shuffle=train_shuffle,
                sampler=train_sampler,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                prefetch_factor=4 if num_workers > 0 else None,
            )

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=2,
                                     drop_last=False, pin_memory=True,
                                     persistent_workers=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train(resume_step=current_step)
    print('Number of pamrams Qformer: ',count_parameters(model.netG))
    print('Number of pamrams SpecMambanet: ',count_parameters(model.SpecMamba))

    # Best-metric tracking, persisted so a resume doesn't clobber best_*.pth
    # with a worse model on the first eval after restart.
    import json
    best_metric_path = os.path.join(opt['path']['models'], 'best_metric.json')
    best_psnr = 0.0
    best_step = 0
    if current_step > 0 and os.path.isfile(best_metric_path):
        try:
            with open(best_metric_path) as f:
                _bm = json.load(f)
            best_psnr = float(_bm.get('best_psnr', 0.0))
            best_step = int(_bm.get('best_step', 0))
            if opt['rank'] == 0:
                logger.info(f'Resumed best PSNR={best_psnr:.2f}dB @ {best_step:,}')
        except Exception as e:
            print(f'Could not read {best_metric_path}: {e}')

    #if opt['rank'] == 0:
    #    logger.info(model.info_network())
    #   logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''


    start_epoch = current_step // len(train_loader)
    for epoch in range(start_epoch, 1000000):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for i, train_data in enumerate(train_loader):

            current_step += 1
            #print(train_data['L'].shape)
            #print(train_data['L'][1])


            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            #print(train_data.shape)
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)
            
            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing  (rank-0 only, sub-sampled, image-saving optional)
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['dist']:
                torch.distributed.barrier()
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                eval_cfg = opt['train'].get('eval', {}) or {}
                eval_max = int(eval_cfg.get('max_images', 64))            # cap eval cost
                eval_save = bool(eval_cfg.get('save_images', False))      # off by default

                torch.cuda.empty_cache()

                avg_psnr = 0.0
                avg_ssim = 0.0
                idx = 0

                for test_data in test_loader:
                    if idx >= eval_max:
                        break
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, _ = os.path.splitext(image_name_ext)

                    if idx % 200 == 0:
                        torch.cuda.empty_cache()
                    model.feed_data(test_data)
                    model.test()
                    visuals = model.current_visuals()

                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border, test_y_channel=False)
                    current_ssim = util.calculate_ssim(E_img, H_img, border=0, scale=1)
                    avg_psnr += current_psnr
                    avg_ssim += current_ssim

                    if eval_save:
                        M_img = util.tensor2uint(visuals['M'])
                        img_dir = os.path.join(opt['path']['images'], img_name)
                        util.mkdir(img_dir)
                        util.imsave(E_img, os.path.join(img_dir, f'{img_name}_{current_step}.png'))
                        util.imsave(M_img, os.path.join(img_dir, f'Mask_{img_name}_{current_step}.png'))

                if idx > 0:
                    avg_psnr /= idx
                    avg_ssim /= idx
                    is_best = avg_psnr > best_psnr
                    if is_best:
                        best_psnr = avg_psnr
                        best_step = current_step
                        model.save('best')
                        with open(best_metric_path, 'w') as f:
                            json.dump({'best_psnr': best_psnr, 'best_step': best_step}, f)
                    logger.info(
                        f' Eval (n={idx}): PSNR={avg_psnr:.2f}dB  SSIM={avg_ssim:.4f}'
                        f'  {"<-- best" if is_best else f"(best={best_psnr:.2f}dB @ {best_step:,})"}\n'
                    )
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['dist']:
                torch.distributed.barrier()


if __name__ == '__main__':
    main()