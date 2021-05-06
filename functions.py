from skimage.color import rgb2lab, lab2rgb
import numpy as np
import torch

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def lab2rgb_tensor(x):
    b, c, h, w = x.shape
    result = []
    for i in range(b):
        data = x[i, :, :, :] # torch.Size([3, 256, 256])
        data = data.permute(1, 2, 0) #torch.Size([256, 256, 3])
        data = data.cpu().numpy()
        data = (data * [100, 255, 255]) - [0, 128, 128]
        # data = data.astype(np.uint8)
        data = lab2rgb(data)
        data = torch.from_numpy(data)
        data = data.permute(2, 0, 1)
        # data = torch.from_numpy(data/255.)
        result.append(data)
    result = torch.stack(result)
    return result
