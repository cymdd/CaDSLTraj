from config import get_parser
from data_loader import Data_loader
from main_cadsl import os,MainCaDSL
from utils import load_arg,save_arg
import torch

def test(args,parser):
    test_set = args.test_set
    #-----create a storage folder-----
    model_dirName = f'test_{test_set}'
    args.model_filepath = os.path.join(args.checkpoint_dir, model_dirName)
    if not os.path.exists(args.model_filepath):
        print("The model parameter file does not exist！")
    #-----load/create the configuration fil----
    args.config = args.model_filepath+'/config_'+args.phase+'.yaml'
    if not load_arg(args,parser):
        save_arg(args)
    args = load_arg(args,parser)
    Dataloader = Data_loader(args, phase="test")
    main_cadsl = MainCaDSL(args, Dataloader)
    main_cadsl.playtest()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.using_cuda:
        torch.cuda.set_device(args.gpu)
    checkpoint_dir = "./checkpoints/"
    args.checkpoint_dir = checkpoint_dir
    if os.path.exists(checkpoint_dir) is False:
        os.makedirs(checkpoint_dir)
    test(args,parser)