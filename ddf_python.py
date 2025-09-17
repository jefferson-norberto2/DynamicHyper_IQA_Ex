import torch
import torch.nn.functional as F

def ddf(x, channel_filter, spatial_filter,
        kernel_size=3, dilation=1, stride=1, kernel_combine="mul"):
    """
    PyTorch fallback para o operador CUDA ddf_mul_ext.

    Args:
        x: Tensor [B, C, H, W]
        channel_filter: Tensor [B, C, K, K]
        spatial_filter: Tensor [B, 1, H_out, W_out]
        kernel_size: int
        dilation: int
        stride: int
        kernel_combine: str, {'mul', 'add'}

    Return:
        out: Tensor [B, C, H_out, W_out]
    """

    B, C, H, W = x.shape
    _, _, K, _ = channel_filter.shape

    # Saída espacial
    H_out = (H + 2*0 - dilation*(K-1) - 1) // stride + 1
    W_out = (W + 2*0 - dilation*(K-1) - 1) // stride + 1

    # 1) Extrai patches da entrada
    patches = F.unfold(x, kernel_size=K, dilation=dilation, stride=stride)
    # patches: [B, C*K*K, H_out*W_out]

    patches = patches.view(B, C, K, K, H_out, W_out)

    # 2) Aplica filtro de canal (um kernel por canal)
    channel_filter = channel_filter.unsqueeze(-1).unsqueeze(-1)  # [B, C, K, K, 1, 1]
    feat = (patches * channel_filter).sum(dim=(2, 3))  # [B, C, H_out, W_out]

    # 3) Aplica filtro espacial
    if spatial_filter.shape[-2:] != (H_out, W_out):
        spatial_filter = F.interpolate(
            spatial_filter, size=(H_out, W_out), mode="nearest"
        )

    # 3) Aplica filtro espacial
    if kernel_combine == "mul":
        out = feat * spatial_filter  # broadcasting: [B, C, H_out, W_out]
    elif kernel_combine == "add":
        out = feat + spatial_filter
    else:
        raise ValueError(f"kernel_combine {kernel_combine} não suportado")

    return out


B, C, H, W = 2, 8, 1024, 768
K = 3

x = torch.randn(B, C, H, W)
channel_filter = torch.randn(B, C, K, K)
spatial_filter = torch.randn(B, 1, H//1, W//1)

out = ddf(x, channel_filter, spatial_filter, kernel_size=K, stride=1, dilation=1)
print(out.shape)  # [2, 8, 16, 16]
