import torch
from torch.nn.functional import affine_grid, grid_sample
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def q_sampling(img, q_mode='q0', op_mode='down'):
    img = img.to(device)
    h, w = img.shape[2], img.shape[3]
    pad = torch.nn.ReflectionPad2d((w // 2, w // 2, h // 2, h // 2))
    img = pad(img)

    if q_mode == 'q0' and op_mode == 'down':
        q = torch.tensor([[1, -1, 0], [1, 1, 0]])
    elif q_mode == 'q1' and op_mode == 'down':
        q = torch.tensor([[1, 1, 0], [-1, 1, 0]])
    elif q_mode == 'q0' and op_mode == 'up':
        q = torch.tensor([[0.5, 0.5, 0], [-0.5, 0.5, 0]])
    elif q_mode == 'q1' and op_mode == 'up':
        q = torch.tensor([[0.5, -0.5, 0], [0.5, 0.5, 0]])
    else:
        raise NotImplementedError("Not available q type")

    q = q[None, ...].type(torch.FloatTensor).repeat(img.shape[0], 1, 1)
    grid = affine_grid(q, img.size(), align_corners=True).type(torch.FloatTensor).to(device)
    img = grid_sample(img, grid, align_corners=True)

    h, w = img.shape[2], img.shape[3]
    img = img[:, :, h // 4:3 * h // 4, w // 4:3 * w // 4]
    return img