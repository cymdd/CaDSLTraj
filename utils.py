import torch
import copy
import numpy as np
import os
import yaml

def L2forTest(outputs,targets,obs_length):
    seq_length = outputs.shape[0]
    error = torch.norm(outputs-targets,p=2,dim=2)
    error_pred_length = error[obs_length-1:]
    error = torch.sum(error_pred_length)
    error_cnt = error_pred_length.numel()
    if error == 0:
        return 0,0,0,0,0,0
    final_error = torch.sum(error_pred_length[-1])
    final_error_cnt = error_pred_length[-1].numel()
    first_erro = torch.sum(error_pred_length[0])
    first_erro_cnt = error_pred_length[0].numel()
    return error.item(),error_cnt,final_error.item(),final_error_cnt,first_erro.item(),first_erro_cnt

def display_performance(perf_dict):
    print("==> Current Performances (ADE & FDE):")
    for a, b in perf_dict.items():
        c = copy.deepcopy(b)
        if isinstance(c, list):
            c[0] = np.round(c[0], 4)
            c[1] = np.round(c[1], 4)
        print("   ", a, c)

def load_arg(p, parser):
    # save arg
    if os.path.exists(p.config):
        with open(p.config, 'r') as f:
            # default_arg = yaml.load(f,Loader=yaml.FullLoader)
            default_arg = yaml.safe_load(f)

        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s=1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False

def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_filepath):
        os.makedirs(args.model_filepath)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)


def modifyArgsfile(file_path, key, value):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    cluster_num_line = None
    for i, line in enumerate(lines):
        if line.startswith(key):
            cluster_num_line = i
            break

    if cluster_num_line is not None:
        del lines[cluster_num_line]

    new_line = f'{key}: {value}\n'
    lines.insert(cluster_num_line, new_line)

    with open(file_path, 'w') as file:
        file.writelines(lines)

    print(f'{key} has been updated to {value}')