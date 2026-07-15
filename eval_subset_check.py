"""Decisive diagnostic: is the full-set 29.8 the same model as the '32.9@n=100'?
Evaluates best_ema on the test set in the SAME order as training, printing the
running-mean PSNR at n=100, n=200, n=500, and full. Single-GPU."""
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import utils_image as util
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.specmamba import Specmamba

opt = option.parse("options/train_unified_SD2_v2.json", is_train=True)
mcfg = opt["model_args"]
dev = "cuda"

test_set = define_Dataset(opt["datasets"]["test"])
loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

net = Specmamba(n_channels=3, n_classes=2, base_dim=int(mcfg["base_dim"]),
                depths=tuple(mcfg["depths"]), shared_scan=True, prompt_dim=int(mcfg["prompt_dim"]),
                input_range="01", stem_stride=int(mcfg["stem_stride"]), lite_expansion=float(mcfg["lite_expansion"]),
                use_mamba_decoder=False, use_mamba_stage3=True, dual_head=True,
                moe_experts=int(mcfg["moe_experts"])).to(dev)
sd = torch.load("Training_logs/Unified_R2MoE_SD2_v2/models/best_ema.pth", map_location=dev)
net.load_state_dict(sd, strict=True); net.eval()

marks = [100, 200, 500]
psnr_sum = 0.0; n = 0
with torch.no_grad():
    for td in loader:
        Lt = td["L"].to(dev); Ht = td["H"].to(dev)[:, 1:]
        _, D = net(Lt, return_aux=False)
        p = util.calculate_psnr(util.tensor2uint(D[0].float().cpu()),
                                util.tensor2uint(Ht[0].float().cpu()), border=1, test_y_channel=False)
        psnr_sum += p; n += 1
        if n in marks:
            print(f"  mean PSNR @ n={n:5d}: {psnr_sum/n:.2f} dB")
print(f"  mean PSNR @ n={n:5d} (FULL): {psnr_sum/n:.2f} dB")
