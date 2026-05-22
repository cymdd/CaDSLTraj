from config import get_parser
from data_loader import Data_loader
from main_cadsl import MainCaDSL, os
from utils import load_arg,save_arg
import torch

def train(args,parser):
    test_set = args.test_set
    #-----create a storage folder-----
    model_dirName = f'test_{test_set}'
    args.model_filepath = os.path.join(args.checkpoint_dir, model_dirName)
    if not os.path.exists(args.model_filepath):
        os.makedirs(args.model_filepath)
        print(f"folder '{model_dirName}' has been created in '{args.checkpoint_dir}'")
    #-----load/create the configuration file----
    args.phase = 'train'
    args.config = args.model_filepath+'/config_'+args.phase+'.yaml'
    if not load_arg(args,parser):
        save_arg(args)
    args = load_arg(args,parser)
    Dataloader = Data_loader(args, phase="train")
    main_cadsl = MainCaDSL(args, Dataloader)
    main_cadsl.playEntireTrain()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.using_cuda:
        torch.cuda.set_device(args.gpu)
    checkpoint_dir = "./checkpoints/"
    args.checkpoint_dir = checkpoint_dir
    if os.path.exists(checkpoint_dir) is False:
        os.makedirs(checkpoint_dir)
    train(args,parser)
