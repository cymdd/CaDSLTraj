import argparse
import ast

def get_parser():
    parser = argparse.ArgumentParser(description='PMIra')
    parser.add_argument('--ifvalid',default=True,type=ast.literal_eval,help="=False,usell train set to train,"
                                                                            "=True,use train set to train and valid")
    parser.add_argument("--final_mode",default=20,type=int)
    parser.add_argument("--T_max",default=1000,type=int)
    parser.add_argument("--eta_min",default=1e-4,type=float)
    parser.add_argument('--output_size',default=2,type=int)
    parser.add_argument('--input_size',default=2,type=int)
    parser.add_argument('--min_obs',default=8,type=int)
    parser.add_argument('--ifGaussian',default=False,type=ast.literal_eval)
    parser.add_argument('--num_pred', type=int, default=1, help='This is the number of predictions for each agent')
    parser.add_argument('--hidden_size', type=int, default=64, help='The size of hidden state')
    parser.add_argument('--gpu', default=0,type=int,help='gpu id')
    parser.add_argument('--using_cuda',default=True,type=ast.literal_eval)
    parser.add_argument('--phase', default='test',help='Set this value to \'train\' or \'test\'')
    parser.add_argument('--GT',default=True)
    parser.add_argument('--train_model', default='CaDSLTraj',help='Your model name')
    parser.add_argument('--load_model', default=-1,type=int,help="load model weights from this index before training or testing")
    parser.add_argument('--model', default='models.CaDSLTraj')
    ######################################
    # -----necessary to provide ----begin
    parser.add_argument('--dataset', type=str, default='ethucy', choices=['ethucy', 'madras'])
    # ethucy = ['eth', 'hotel', 'zara01', 'zara02', 'students001', 'students003', 'uni_examples', 'zara03']
    # madras = ['TopView_1B','TopView_2A','TopView_2B','TopView_2C']
    parser.add_argument('--test_set', type=str, default='eth', help='chose test set')
    # -----necessary to provide ----end
    ######################################
    parser.add_argument('--config')
    parser.add_argument('--val_fraction',default=0.2,type=float)
    parser.add_argument('--test_fraction',default=0.1,type=float)
    #model
    parser.add_argument('--seq_length',default=20,type=int)
    parser.add_argument('--obs_length',default=8,type=int)
    parser.add_argument('--pred_length',default=12,type=int)
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--num_epochs',default=1000,type=int)
    parser.add_argument('--randomRotate',default=True,type=ast.literal_eval,help="=True:random rotation of each trajectory fragment")
    parser.add_argument('--neighbor_dist',default=10,type=int, help='neighborhood distance threshold')
    parser.add_argument('--learning_rate',default=1e-03,type=float)
    parser.add_argument('--clip',default=10,type=int)
    parser.add_argument('--eff_nei_thre', default=0.1, type=float, help='effective neighbor threshold')
    parser.add_argument('--lambda_loss', default=0.1, type=float,help='loss balance parameter')
    parser.add_argument('--grid_size', default=16, type=int,help='grid size')
    parser.add_argument('--num_heads', default=4, type=int,help='number of attention heads')
    parser.add_argument('--num_samples', default=20, type=int)
    return parser