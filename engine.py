import torch
import torch.optim as optim
from model import *
import util


class trainer():
    def __init__(self, no_pmt, prompt_dim, scaler, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device,
                 supports, gcn_bool, addaptadj, aptinit, remove_list, add_list):
        self.remove_list = remove_list
        self.add_list = add_list
        self.no_pmt, self.prompt_dim, self.device, self.num_nodes, self.dropout, self.supports, self.gcn_bool, self.addaptadj, self.aptinit, self.in_dim, self.seq_length, self.nhid = no_pmt, prompt_dim, device, num_nodes, dropout, supports, gcn_bool, addaptadj, aptinit, in_dim, seq_length, nhid
        self.set_model()
        self.lrate = lrate
        self.wdecay = wdecay
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        self.use_prompt = 0
        self.mean = 0
        self.var = 0
        self.add_test = 0
        self.remove_test = 0

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input, self.use_prompt)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        # real = torch.unsqueeze(real_val,dim=1)
        real = real_val
        predict = self.scaler.inverse_transform(output)
        if self.remove_list is not None:
            # print(predict.shape,real.shape) torch.Size([64, 1, 184, 12]) torch.Size([64, 184, 12])
            predict[:, :, self.remove_list, :] = 0
            real[:, self.remove_list, :] = 0
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        total_gradient_sum = 0.0

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()

        # 计算残差
        residual = real - torch.squeeze(predict)
        residual = residual.flatten()

        return loss.item(), mape, rmse, residual

    def eval(self, input, real_val):
        self.model.eval()

        # 使用 torch.no_grad() 来阻止梯度计算
        with torch.no_grad():
            # 假设 input 已经是一个适当的张量，不需要额外的 padding
            output = self.model(input, self.use_prompt, add_test=self.add_test, remove_test=self.remove_test)
            output = output.transpose(1, 3)

            # real_val 不需要梯度，因为它只用于计算损失和评估指标
            real = torch.unsqueeze(real_val, dim=1)

            # 逆变换输出以进行预测
            predict = self.scaler.inverse_transform(output)

            if self.add_list is not None:
                predict[:, :, self.add_list, :] = 1
                real[:, :, self.add_list, :] = 1

            # 计算损失和评估指标
            loss = self.loss(predict, real, 0.0)
            mape = util.masked_mape(predict, real, 0.0).item()
            rmse = util.masked_rmse(predict, real, 0.0).item()

            # 计算残差
            residual = real - predict
            residual = residual.flatten()

        return loss.item(), mape, rmse, residual

    def set_model(self):
        self.model = gwnet(self.no_pmt, self.prompt_dim, self.device, self.remove_list, self.add_list, self.num_nodes,
                           self.dropout, supports=self.supports, gcn_bool=self.gcn_bool, addaptadj=self.addaptadj,
                           aptinit=self.aptinit, in_dim=self.in_dim, out_dim=self.seq_length,
                           residual_channels=self.nhid, dilation_channels=self.nhid, skip_channels=self.nhid * 8,
                           end_channels=self.nhid * 16)
        self.model.to(self.device)

    def set_opt(self, retrain=False):
        if retrain:
            learning_rate = self.lrate
        else:
            learning_rate = self.optimizer.param_groups[0]['lr']
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=self.wdecay)

    def train_prompt_t(self, input, real_val):
        self.model.pemb_multi_t.train()
        self.optimizer.zero_grad()
        prompt = input[:, :, :, :12].squeeze()
        # output -- reconstruction
        reconstruction, _1 = self.model.pemb_multi_t(prompt)

        # reconstruction loss
        loss = self.loss(reconstruction, prompt, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.pemb_multi_t.parameters(), self.clip)
        self.optimizer.step()
        return loss.item()

    def train_prompt_s(self, input, real_val):
        self.model.pemb_multi_s.train()
        self.optimizer.zero_grad()
        prompt = input[:, :, :, :12].squeeze()
        # output -- reconstruction
        loss, _2 = self.model.pemb_multi_s(prompt)

        # reconstruction loss
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.pemb_multi_s.parameters(), self.clip)
        self.optimizer.step()
        return loss.item()
