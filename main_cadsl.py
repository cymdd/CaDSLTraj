from utils import *
import torch
import time
import torch.nn as nn
import os
import pickle
from tqdm import tqdm


class MainCaDSL:
    def __init__(self, args, Dataloader):
        self.args = args
        self.lr = self.args.learning_rate
        self.dataloader_gt = Dataloader
        self.epoch = 0

    # ── 模型保存/加载 ──────────────────────────────────────

    def save_model(self, epoch):
        model_path = self.args.model_filepath + '/' + self.args.train_model + '_' + str(epoch) + '.pth'
        checkpoint = {
            'epoch': epoch,
            'net': self.net,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler
        }
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
        self.args.load_model = epoch
        modifyArgsfile(self.args.config, 'load_model', epoch)

    def load_model(self):
        if self.args.load_model >= 0:
            self.args.model_save_path = (self.args.model_filepath + '/' +
                                         self.args.train_model + '_' +
                                         str(self.args.load_model) + '.pth')
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(
                    self.args.model_save_path,
                    map_location={'cuda:0': 'cuda:' + str(self.args.gpu)},
                    weights_only=False
                )
                self.epoch = int(checkpoint['epoch']) + 1
                self.net = checkpoint['net']
                self.optimizer = checkpoint['optimizer']
                self.scheduler = checkpoint['scheduler']
                print('Loaded checkpoint at epoch', checkpoint['epoch'])

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss(reduce=False)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.args.num_epochs,
            eta_min=self.args.eta_min
        )

    # ── 输入数据准备 ────────────────────────────────────────

    def get_inputsfw(self, batch, epoch, isval):

        if isval:
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_val_batch(batch, epoch)
        else:
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_train_batch(batch, epoch)

        inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
        if self.args.using_cuda:
            inputs_gt = tuple([i.cuda() for i in inputs_gt])

        batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt

        # CaDSL-Traj 输入格式: [N, T, 2]
        if batch_abs_gt.dim() == 3 and batch_abs_gt.shape[0] != batch_abs_gt.shape[1]:
            if batch_abs_gt.shape[0] == self.args.seq_length:
                batch_abs_gt = batch_abs_gt.permute(1, 0, 2)   # [N, T, 2]
                batch_norm_gt = batch_norm_gt.permute(1, 0, 2)  # [N, T, 2]

        inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split
        return inputs_fw, batch_norm_gt

    # ── 测试 ────────────────────────────────────────────────

    def playtest(self):
        print('Testing begin')
        model_filepath = os.path.join(self.args.model_filepath, "best_model.pth")
        if os.path.exists(model_filepath):
            model_state_dict = torch.load(
                model_filepath,
                map_location={'cuda:1': 'cuda:' + str(self.args.gpu)}
            )
            self.net = model_state_dict
            self.net.args = self.args
            test_error, test_final_error, first_erro_test, t = self.test_epoch()
            print('test_error: {:.5f} test_final_error: {:.5f} time: {:.5f}'.format(
                test_error, test_final_error, t))
        else:
            print("No model weight file!")

    # ── 完整训练流程 ─────────────────────────────────────────

    def playEntireTrain(self):
        torch.cuda.empty_cache()
        perf_dict = {"whole_model[ADE,FDE]": [1e3, 1e3]}
        dict_key = "whole_model[ADE,FDE]"

        # 导入并实例化 CaDSL-Traj 模型
        from model import CaDSLTraj
        self.net = CaDSLTraj(self.args)
        self.set_optimizer()

        if self.args.load_model >= 0:
            self.load_model()
        else:
            self.epoch = 0
            if self.args.using_cuda:
                self.net = self.net.cuda()

        print('CaDSL-Traj Training begin')
        epochs_tqdm = tqdm(range(self.epoch, self.args.num_epochs))
        start = time.time()

        for epoch in epochs_tqdm:
            with torch.no_grad():
                print('Epoch-{0} lr: {1}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            train_loss = self.train_epoch(epoch)
            # train_loss=0
            val_error, val_final_error, val_first, valtime = self.val_epoch()

            with torch.no_grad():
                val_loss_logfilepath = os.path.join(self.args.model_filepath, 'val_loss_log.txt')
                content = (f'epoch={epoch+1} | valid_error={val_error:.5f} '
                           f'| valid_final={val_final_error:.5f} | time={valtime:.2f}')
                with open(val_loss_logfilepath, 'a') as f:
                    f.write(content + '\n')

            self.scheduler.step()
            if epoch % 10 == 0:
                self.save_model(epoch + 1)

            with torch.no_grad():
                print('---- epoch {} | train_loss={:.5f} | valid_ADE={:.3f} | valid_FDE={:.3f}'.format(
                    epoch, train_loss, val_error, val_final_error))
                if val_final_error < perf_dict[dict_key][1]:
                    perf_dict[dict_key] = [val_error, val_final_error]
                    torch.save(self.net, os.path.join(self.args.model_filepath, "best_model.pth"))
                    with open(os.path.join(self.args.model_filepath, "Performances.pkl"), "wb") as f:
                        pickle.dump(perf_dict, f, 4)
                    print("==> best_model Saved")

            epochs_tqdm.update(1)

        end = time.time()
        print(f'train_time = {(end - start) / 3600:.2f} H')
        epochs_tqdm.close()

    # ── 训练 epoch ──────────────────────────────────────────

    def train_epoch(self, epoch):
        self.net.train()
        loss_epoch = 0

        for batch in range(self.dataloader_gt.trainbatchnums):
            start = time.time()
            inputs_fw, batch_norm_gt = self.get_inputsfw(batch, epoch, isval=False)

            self.net.zero_grad()
            total_loss, _ = self.net.forward(inputs_fw, batch, iftest=False)

            loss_epoch += total_loss.item()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()

            end = time.time()
            with torch.no_grad():
                print('train-{}/{} (epoch {}), loss={:.5f}, time={:.3f}s'.format(
                    batch + 1, self.dataloader_gt.trainbatchnums, epoch,
                    total_loss.item(), end - start))

        return loss_epoch / (self.dataloader_gt.trainbatchnums + 1e-8)

    # ── 验证 epoch ──────────────────────────────────────────

    def val_epoch(self):
        self.net.eval()
        error_epoch,final_error_epoch, first_erro_epoch = 0,0,0
        error_epoch_list, final_error_epoch_list, first_erro_epoch_list= [], [], []
        error_cnt_epoch, final_error_cnt_epoch, first_erro_cnt_epoch = 1e-5,1e-5,1e-5
        start = time.time()

        for batch in range(self.dataloader_gt.valbatchnums):
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_val_batch(batch, None)
            inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
            if self.args.using_cuda:
                inputs_gt = tuple([i.cuda() for i in inputs_gt])

            batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
            #
            # # 维度转换 [T, N, 2] → [N, T, 2]
            if batch_abs_gt.shape[0] == self.args.seq_length:
                batch_abs_gt = batch_abs_gt.permute(1, 0, 2)
                # batch_norm_gt = batch_norm_gt.permute(1, 0, 2)

            inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split
            _, full_pre_tra = self.net.forward(inputs_fw, batch, iftest=True)
            batch_abs_gt = batch_abs_gt.permute(1, 0, 2)
            for pre_tra in full_pre_tra:
                error, error_cnt, final_error, final_error_cnt, first_erro, first_erro_cnt = \
                    L2forTest(pre_tra, batch_abs_gt[:, :, :2], self.args.obs_length)
                error_epoch_list.append(error)
                final_error_epoch_list.append(final_error)
                first_erro_epoch_list.append(first_erro)

            first_erro_epoch += min(first_erro_epoch_list)
            final_error_epoch += min(final_error_epoch_list)
            error_epoch += min(error_epoch_list)
            error_cnt_epoch += error_cnt
            final_error_cnt_epoch += final_error_cnt
            first_erro_cnt_epoch += first_erro_cnt
            error_epoch_list, final_error_epoch_list, first_erro_epoch_list = [], [], []
        end = time.time()
        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch, first_erro_epoch / first_erro_cnt_epoch,end-start

    # ── 测试 epoch ──────────────────────────────────────────

    def test_epoch(self):
        self.net.eval()
        error_epoch, final_error_epoch, first_erro_epoch = 0, 0, 0
        error_epoch_list, final_error_epoch_list, first_erro_epoch_list = [], [], []
        error_cnt_epoch, final_error_cnt_epoch, first_erro_cnt_epoch = 1e-5, 1e-5, 1e-5
        start = time.time()
        
        for batch in range(self.dataloader_gt.testbatchnums):
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_test_batch(batch, None)
            inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
            if self.args.using_cuda:
                inputs_gt = tuple([i.cuda() for i in inputs_gt])
        
            batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
            #
            # # 维度转换 [T, N, 2] → [N, T, 2]
            if batch_abs_gt.shape[0] == self.args.seq_length:
                batch_abs_gt = batch_abs_gt.permute(1, 0, 2)
                # batch_norm_gt = batch_norm_gt.permute(1, 0, 2)
        
            inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split
            _, full_pre_tra = self.net.forward(inputs_fw, batch, iftest=True)
            batch_abs_gt = batch_abs_gt.permute(1, 0, 2)
            for pre_tra in full_pre_tra:
                error, error_cnt, final_error, final_error_cnt, first_erro, first_erro_cnt = \
                    L2forTest(pre_tra, batch_abs_gt[:, :, :2], self.args.obs_length)
                error_epoch_list.append(error)
                final_error_epoch_list.append(final_error)
                first_erro_epoch_list.append(first_erro)
        
            first_erro_epoch += min(first_erro_epoch_list)
            final_error_epoch += min(final_error_epoch_list)
            error_epoch += min(error_epoch_list)
            error_cnt_epoch += error_cnt
            final_error_cnt_epoch += final_error_cnt
            first_erro_cnt_epoch += first_erro_cnt
            error_epoch_list, final_error_epoch_list, first_erro_epoch_list = [], [], []
        end = time.time()
        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch, first_erro_epoch / first_erro_cnt_epoch, end - start

