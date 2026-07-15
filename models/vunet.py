import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels, eps=1e-6),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels, eps=1e-6),
            nn.Dropout2d(p=0.2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels, eps=1e-6),
            nn.Dropout2d(p=0.2)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=False)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out

class VDrawBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin_mean = nn.Linear(in_features=in_features, out_features=out_features)
        self.lin_var = nn.Linear(in_features=in_features, out_features=out_features)

    # Reparameterisation trick
    def forward(self, x):
        z_mean = self.lin_mean(x)
        z_var = self.lin_var(x)

        #if self.training:
        #    std = torch.exp(0.5 * z_var)
        #    epsilon = torch.empty_like(z_var, device=z_mean.device).normal_()
        #    x_out = z_mean + std * epsilon
        #else:
        #    x_out = z_mean
        std = torch.exp(0.5 * z_var)
        epsilon = torch.empty_like(z_var, device=z_mean.device).normal_()
        x_out = z_mean + std * epsilon


        return x_out, z_mean, z_var



class Variable_Unet_512(nn.Module):

    def __init__(self, n_classes, img_ch=3):
        super(Variable_Unet_512, self).__init__()
        self.n_channels = img_ch
        self.n_classes = n_classes
        self.bilinear = True

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 128)
        self.Conv4 = ConvBlock(128, 256)
        self.Conv5 = ConvBlock(256, 256)
        self.Conv6 = ConvBlock(256, 512)
        self.Conv7 = ConvBlock(512, 512)
        self.Conv7 = ConvBlock(512, 512)
        self.Conv8 = ConvBlock(512, 1024)
        self.Conv9 = ConvBlock(1024, 1024)

        self.var = VDrawBlock(1024 * 2 * 2, 1024 * 2 * 2)

        self.Up9 = UpConv(1024, 1024)
        self.Att9 = AttentionBlock(F_g=1024, F_l=1024, n_coefficients=1024)
        self.UpConv9 = ConvBlock(2048, 1024)

        self.Up8 = UpConv(1024, 512)
        self.Att8 = AttentionBlock(F_g=512, F_l=512, n_coefficients=512)
        self.UpConv8 = ConvBlock(1024, 512)

        self.Up7 = UpConv(512, 512)
        self.Att7 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv7 = ConvBlock(1024, 512)

        self.Up6 = UpConv(512, 256)
        self.Att6 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv6 = ConvBlock(512, 256)

        self.Up5 = UpConv(256, 256)
        self.Att5 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv5 = ConvBlock(512, 256)

        self.Up4 = UpConv(256, 128)
        self.Att4 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv4 = ConvBlock(256, 128)

        self.Up3 = UpConv(128, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        N, C, H, W = x.shape
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        e6 = self.MaxPool(e5)
        e6 = self.Conv6(e6)

        e7 = self.MaxPool(e6)
        e7 = self.Conv7(e7)

        e8 = self.MaxPool(e7)
        e8 = self.Conv8(e8)

        e9 = self.MaxPool(e8)
        e9 = self.Conv9(e9)

        x111 = torch.flatten(e9, start_dim=1)
        x111, z_mean, z_logvar = self.var(x111)
        x111 = x111.view(N, -1, 2, 2)

        d9 = self.Up9(x111)
        s8 = self.Att9(gate=d9, skip_connection=e8)
        d9 = torch.cat((s8, d9), dim=1)  # concatenate attention-weighted skip connection with previous layer output
        d9 = self.UpConv9(d9)

        d8 = self.Up8(d9)
        s7 = self.Att8(gate=d8, skip_connection=e7)
        d8 = torch.cat((s7, d8), dim=1)  # concatenate attention-weighted skip connection with previous layer output
        d8 = self.UpConv8(d8)

        d7 = self.Up7(d8)
        s6 = self.Att7(gate=d7, skip_connection=e6)
        d7 = torch.cat((s6, d7), dim=1)  # concatenate attention-weighted skip connection with previous layer output
        d7 = self.UpConv7(d7)

        d6 = self.Up6(d7)
        s5 = self.Att6(gate=d6, skip_connection=e5)
        d6 = torch.cat((s5, d6), dim=1)  # concatenate attention-weighted skip connection with previous layer output
        d6 = self.UpConv6(d6)

        d5 = self.Up5(d6)
        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1)  # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)

        return out, z_mean, z_logvar




class Variable_Unet_light(nn.Module):

    def __init__(self, n_classes, img_ch=3):
        super(Variable_Unet_light, self).__init__()
        self.n_channels = img_ch
        self.n_classes = n_classes
        self.bilinear = True

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 32)
        self.Conv2 = ConvBlock(32, 64)
        self.Conv3 = ConvBlock(64, 64)
        self.Conv4 = ConvBlock(64, 128)
        self.Conv5 = ConvBlock(128, 128)
        self.Conv6 = ConvBlock(128, 256)
        self.Conv7 = ConvBlock(256, 256)
        self.Conv7 = ConvBlock(256, 256)
        self.Conv8 = ConvBlock(256, 512)
        self.Conv9 = ConvBlock(512, 512)

        self.var = VDrawBlock(512 * 2 * 2, 512 * 2 * 2)

        self.Up9 = UpConv(512, 512)
        self.Att9 = AttentionBlock(F_g=512, F_l=512, n_coefficients=512)
        self.UpConv9 = ConvBlock(1024, 512)

        self.Up8 = UpConv(512, 256)
        self.Att8 = AttentionBlock(F_g=256, F_l=256, n_coefficients=256)
        self.UpConv8 = ConvBlock(512, 256)

        self.Up7 = UpConv(256, 256)
        self.Att7 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv7 = ConvBlock(512, 256)

        self.Up6 = UpConv(256, 128)
        self.Att6 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv6 = ConvBlock(256, 128)

        self.Up5 = UpConv(128, 128)
        self.Att5 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv5 = ConvBlock(256, 128)

        self.Up4 = UpConv(128, 64)
        self.Att4 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv4 = ConvBlock(128, 64)

        self.Up3 = UpConv(64, 64)
        self.Att3 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv3 = ConvBlock(128, 64)

        self.Up2 = UpConv(64, 32)
        self.Att2 = AttentionBlock(F_g=32, F_l=32, n_coefficients=32)
        self.UpConv2 = ConvBlock(64, 32)

        self.Conv = nn.Conv2d(32, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        N, C, H, W = x.shape
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        e6 = self.MaxPool(e5)
        e6 = self.Conv6(e6)

        e7 = self.MaxPool(e6)
        e7 = self.Conv7(e7)

        e8 = self.MaxPool(e7)
        e8 = self.Conv8(e8)

        e9 = self.MaxPool(e8)
        e9 = self.Conv9(e9)

        x111 = torch.flatten(e9, start_dim=1)
        x111, z_mean, z_logvar = self.var(x111)
        x111 = x111.view(N, -1, 2, 2)

        d9 = self.Up9(x111)
        s8 = self.Att9(gate=d9, skip_connection=e8)
        d9 = torch.cat((s8, d9), dim=1)  # concatenate attention-weighted skip connection with previous layer output
        d9 = self.UpConv9(d9)

        d8 = self.Up8(d9)
        s7 = self.Att8(gate=d8, skip_connection=e7)
        d8 = torch.cat((s7, d8), dim=1)  # concatenate attention-weighted skip connection with previous layer output
        d8 = self.UpConv8(d8)

        d7 = self.Up7(d8)
        s6 = self.Att7(gate=d7, skip_connection=e6)
        d7 = torch.cat((s6, d7), dim=1)  # concatenate attention-weighted skip connection with previous layer output
        d7 = self.UpConv7(d7)

        d6 = self.Up6(d7)
        s5 = self.Att6(gate=d6, skip_connection=e5)
        d6 = torch.cat((s5, d6), dim=1)  # concatenate attention-weighted skip connection with previous layer output
        d6 = self.UpConv6(d6)

        d5 = self.Up5(d6)
        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1)  # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)

        return out, z_mean, z_logvar





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    net = Variable_Unet_512(n_classes=2).cuda()
    print(count_parameters(net))
    inp = torch.randn(1, 3, 512, 512).cuda()
    net.load_state_dict(torch.load('./weight/VAUNet/SHIQ.pth'))
    out,_,_ = net(inp)
    print(out.shape)



