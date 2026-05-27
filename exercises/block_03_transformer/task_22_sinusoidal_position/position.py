import math

import torch


def sinusoidal_position_encoding(max_len, dim, device=None):
    position = torch.arange(max_len, device=device).float().unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(max_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


if __name__ == "__main__":
    print(sinusoidal_position_encoding(4, 8))

