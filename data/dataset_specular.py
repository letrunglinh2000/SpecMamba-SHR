import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import torch.nn.functional as F
from PIL import Image

class DatasetSpecular(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR Quaternion.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetSpecular, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = 1
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf

        #print("HELLOOOOOOOOOOOOOO")
        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])
        self.paths_M = util.get_image_paths(opt['dataroot_M']) if opt['dataroot_M'] is not None else None
        #print(self.paths_H)
        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))
            if self.paths_M is not None:
                assert len(self.paths_L) == len(self.paths_M), 'L/M mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_M))

        # ------------------------------------
        # Drop spatially-mismatched pairs (train only).
        # The random crop derives its coords from L and applies them to H/M, so a pair
        # whose H/L/M sizes differ (e.g. some RD images) yields variable-size patches
        # and crashes default_collate. Such pairs are also not pixel-aligned, so the GT
        # is wrong -- skip them. Uses PIL lazy .size (header only), so it's cheap.
        # No-op for aligned datasets like SHIQ/SD2.
        if self.opt.get('phase') == 'train' and self.paths_L and self.paths_M is not None:
            def _wh(p):
                try:
                    with Image.open(p) as im:
                        return im.size  # (W, H), header-only, no decode
                except Exception:
                    return None
            keepH, keepL, keepM, dropped = [], [], [], 0
            for hp, lp, mp in zip(self.paths_H, self.paths_L, self.paths_M):
                sh, sl, sm = _wh(hp), _wh(lp), _wh(mp)
                if sh is not None and sh == sl == sm:
                    keepH.append(hp); keepL.append(lp); keepM.append(mp)
                else:
                    dropped += 1
            if dropped:
                print('[DatasetSpecular] skipped {} size-mismatched/unreadable pairs; '
                      'kept {}.'.format(dropped, len(keepH)))
            self.paths_H, self.paths_L, self.paths_M = keepH, keepL, keepM

    def __getitem__(self, index):

        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)

        # ------------------------------------
        # modcrop
        # ------------------------------------
        #img_H = util.modcrop(img_H, self.sf)

        # ------------------------------------
        # get L image
        # ------------------------------------

        L_path = self.paths_L[index]
        img_L = util.imread_uint(L_path, self.n_channels)
        img_L = util.uint2single(img_L)


        # ------------------------------------
        # get M image
        # ------------------------------------
        if self.paths_M is None:
            M_path = ''
            img_M = np.zeros((img_L.shape[0], img_L.shape[1], 1), dtype=np.float32)
            has_mask = 0.0
        else:
            M_path = self.paths_M[index]
            img_M = util.imread_uint(M_path, 1)
            img_M = util.uint2single(img_M)
            # img_M = np.where(img_M >= 10/255.0, 1, 0).astype(np.float32)
            img_M = np.where(img_M >= 0.5, 1, 0).astype(np.float32)

            has_mask = 1.0





        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            img_M = img_M[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]



            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H, img_M = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode), util.augment_img(img_M, mode=mode)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------

        ### resize M to 512

        img_L, img_M = util.single2tensor3(img_L), util.single2tensor3(img_M)

        img_H = util.single2quaternion(img_H)
        if L_path is None:
            L_path = H_path


        return {
            'L': img_L,
            'H': img_H,
            'M': img_M,
            'has_mask': has_mask,
            'L_path': L_path,
            'H_path': H_path,
            'M_path': M_path,
        }

    def __len__(self):
        return len(self.paths_H)
