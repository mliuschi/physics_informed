'''
This code generates the prediction on one instance. 
Both the ground truth and the prediction are saved in a .pt file.
'''
import os, sys
import yaml
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from models import FNO3d
from baselines.pdearena_unet import Unet, FourierUnet
from baselines.pdearena_resnet_fno import ResNet
from baselines.neuralop_fno import FNO3D

# Utils
from baselines.pdearena_resnet_fno import partialclass, FourierBasicBlock

from train_utils.datasets import KFDataset
from train_utils.losses import LpLoss
from train_utils.utils import count_params


@torch.no_grad()
def get_pred(args):
    with open(args.config, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    basedir = os.path.join('exp', config['log']['logdir'])
    save_dir = os.path.join(basedir, 'results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,'prediction_2000.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare data
    dataset = KFDataset(paths=config['data']['paths'], 
                        raw_res=config['data']['raw_res'],
                        data_res=config['data']['data_res'], 
                        pde_res=config['data']['data_res'], 
                        n_samples=config['data']['n_test_samples'],
                        total_samples=config['data']['total_test_samples'],
                        offset=config['data']['testoffset'], 
                        t_duration=config['data']['t_duration'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    # create model
    # model = FNO3d(modes1=config['model']['modes1'],
    #             modes2=config['model']['modes2'],
    #             modes3=config['model']['modes3'],
    #             fc_dim=config['model']['fc_dim'],
    #             layers=config['model']['layers'], 
    #             act=config['model']['act'], 
    #             pad_ratio=config['model']['pad_ratio']).to(device)

    # model = Unet(
    #     n_input_scalar_components = 4,
    #     n_input_vector_components = 0,
    #     n_output_scalar_components = 1,
    #     n_output_vector_components = 0,
    #     time_history = 33,
    #     time_future = 33,
    #     hidden_channels = 64,
    #     activation = 'gelu',
    #     norm = True,                                    # UNet-mod
    #     ch_mults = (1, 2, 2, 4),
    #     is_attn = (False, False, False, False),
    #     mid_attn = False,
    #     n_blocks = 2,
    #     use1x1 = False
    # ).to(device)
    
    model = FourierUnet(                            # U-FNet1-16m
        n_input_scalar_components = 4,
        n_input_vector_components = 0,
        n_output_scalar_components = 1,
        n_output_vector_components = 0,
        time_history = 33,
        time_future = 33,
        hidden_channels = 64,
        activation = "gelu",
        modes1 = 16,
        modes2 = 16,
        norm = True,
        ch_mults = (1, 2, 2, 4),
        is_attn = (False, False, False, False),
        mid_attn = False,
        n_blocks = 2,
        n_fourier_layers = 1,
        mode_scaling = True,
        use1x1 = True,
    ).to(device)

    # model = ResNet(                                    # FNO-128-16m
    #     n_input_scalar_components = 4,
    #     n_input_vector_components = 0,
    #     n_output_scalar_components = 1,
    #     n_output_vector_components = 0,
    #     block = partialclass("CustomFourierBasicBlock", FourierBasicBlock, modes1=16, modes2=16),
    #     num_blocks = [1, 1, 1, 1],
    #     time_history = 33,
    #     time_future = 33,
    #     hidden_channels = 128,
    #     activation = "gelu",
    #     norm = False,
    #     diffmode = False,
    #     usegrid = False,
    # ).to(device)

    # model = FNO3D(
    #     n_modes_height = 16,
    #     n_modes_width = 32,
    #     n_modes_depth = 32,
    #     hidden_channels = 32,
    #     in_dim = 4, 
    #     out_dim = 1,
    #     lifting_channels = 256,
    #     projection_channels = 256,
    #     n_layers = 4,
    #     nonlinearity = F.gelu,
    #     use_mlp = False,
    #     mlp_dropout = 0.0,
    #     mlp_expansion = 0.5,
    #     norm = None
    # ).to(device)

    num_params = count_params(model)
    print(f'Number of parameters: {num_params}')
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % args.ckpt_path)

    # metric
    lploss = LpLoss(size_average=True)
    model.eval()
    truth_list = []
    pred_list = []
    time_list = []
    for u, a_in in dataloader:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        u, a_in = u.to(device), a_in.to(device)

        # Unet only
        u = u[:, ::4, ::4, ::2]
        a_in = a_in[:, ::4, ::4, ::2]

        # # PDE Arena FNO only: FFT in space only; must interpolate in time
        # u = u[:, :, :, ::2]
        # a_in = a_in[:, :, :, ::2]
        
        a_in = a_in.permute(0, 4, 3, 1, 2)   # B, C, T, X, Y

        start.record()

        out = model(a_in)
        
        end.record()
        torch.cuda.synchronize()
        time_list.append(start.elapsed_time(end))

        # Unet only
        if isinstance(model, Unet) or isinstance(model, FNO3D):
            out = out.squeeze(1).permute(0, 2, 3, 1)   # B, X, Y, T
        elif isinstance(model, FourierUnet) or isinstance(model, ResNet):
            out = out.squeeze(2).permute(0, 2, 3, 1)   # B, X, Y, T
        else:
            raise NotImplementedError

        data_loss = lploss(out, u)
        print(data_loss.item())
        truth_list.append(u.cpu())
        pred_list.append(out.cpu())

    truth_arr = torch.cat(truth_list, dim=0)
    pred_arr = torch.cat(pred_list, dim=0)
    print("Mean inference time:", sum(time_list) / len(time_list))
    torch.save({
        'truth': truth_arr,
        'pred': pred_arr,
    }, save_path)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()
    get_pred(args)