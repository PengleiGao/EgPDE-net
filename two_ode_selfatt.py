import os
import numpy as np
import torch
import sys
sys.path.append('../')
import argparse
import torch.nn as nn
from config_hyper_para import *
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_prepro import *
from torch.nn import Parameter
import math

from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_normal


parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', '-a', type=eval, default=False)
parser.add_argument('--dataset', '-d', help="dataset", type=str)
parser.add_argument('--depth', type=int, default=10, help='depth for wide resnet')
parser.add_argument('--num_epochs', type=int, default=100, help='epochs to train the model')
parser.add_argument('--hid_size', type=int, default=10, help='hidden dimension for odenet')
parser.add_argument('--rnn_size', type=int, default=64, help='hidden dimension for rnn')
parser.add_argument('--selfatt_size', type=int, default=30, help='hidden dimension for self-attention')
parser.add_argument('--head', type=int, default=2, help='multi-head for selfatt')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='the batch size to use')
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--gpu', type=str, default=0, help='the gpu to use')
parser.add_argument('--seed', type=int, default=123, help='the seed to control randomness')
parser.add_argument('--gate_type', type=str, default='tensor', help='gate type for tensorized gru')
args = parser.parse_args()

data_name = args.dataset
if 'energy' in data_name:
    data_path = 'raw_data/energydata_complete.csv'
elif 'nasdaq' in data_name:
    data_path = 'raw_data/nasdaq100_padding.csv'
elif 'pm25' in data_name:
    data_path = 'raw_data/pm25_rawdata.csv'
elif 'sml' in data_name:
    data_path = 'raw_data/sml_rawdata.csv'
elif 'elec' in data_name:
    data_path = 'raw_data/electricity_rawdata.csv'
elif 'etth1' in data_name:
    data_path = 'raw_data/ETTh1.csv'
elif 'etth2' in data_name:
    data_path = 'raw_data/ETTh2.csv'
else:
    print("wrong dataset name")

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.train_dir = 'arbitrary-step/' + data_name + '_two_ode'
print('current datset: {}'.format(data_name))
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
target_col = target_col_dic[data_name]
indep_col = indep_col_dic[data_name]
norm = True
win_size = 20
pre_T = 5
#depth = 22
#num_epochs = 100
#hid_size = 5

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

train_share = 0.8
is_stateful = False
normalize_pattern = 2

train_T = torch.linspace(0., 3., 4).to(device)
test_T = torch.Tensor([0., 1., 1.5, 2., 2.5, 3.]).to(device)
criterion = torch.nn.MSELoss()
criterionL1 = torch.nn.L1Loss()
num = 301
path = '/opt/data/private/ExNODE-master/PRODE/' + args.train_dir + '/'
#path = '/home/penglei/MT-GRU/p-ode/arbitrary-step/' + data_name + '/'

best_log_dir = path + str(pre_T) + '_' + data_name + '_' + str(norm) + '_pxz_two_odenet_' + str(num) + '_.pth'

log_err_file = path + str(pre_T) + '_' + data_name + '_' + str(norm) + '_pxz_two_odenet_' + str(num) + '.txt'

class ODE_pzx_trans(nn.Module):
    def __init__(self, hid_size, use_adjoint=False):
        super(ODE_pzx_trans, self).__init__()

        self.ode_func = ODEnet(hid_size)
        self.hid_size = hid_size
        self.use_adjoint = use_adjoint

    def forward(self, z0, t):
        odeint = odeint_adjoint if self.use_adjoint else odeint_normal
        z_t = odeint(self.ode_func, z0, t)
        z_t = F.softmax(z_t, dim=2)

        return z_t


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        d_k = q.size(-1)
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d_k)
        if scale:
            attn = attn * scale
        if attn_mask:
            attn = attn.masked_fill_(attn_mask, -np.inf)
        attn = self.dropout((attn))
        output = torch.matmul(self.softmax(attn), v)
        attn = attn[:, :, -1]
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, driving_size, d_model, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        self.emb = nn.Linear(driving_size, d_model)
        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.out = nn.Linear(d_model, driving_size)

    def forward(self, x, mask=None):
        emb = self.emb(x)
        q, k ,v = emb, emb, emb
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)
        q = self.out(q)

        return q[:, -1, :]


class ODEnet(nn.Module):
    def __init__(self, hid_size):
        super(ODEnet, self).__init__()

        #self.ih = nn.Linear(step, hid_perdim)
        #self.hidden_dim = hid_size * 2
        self.hidden_dim = hid_size
        self.lin_hh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.lin_hz = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.lin_hr = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.out = nn.Linear(self.hidden_dim, hid_size)
        self.nfe = 0

    def forward(self, t, dz):
        self.nfe += 1
        #dz = torch.cat((zy, zx), -1)
        x = torch.zeros_like(dz)
        r = F.sigmoid(x + self.lin_hr(dz))
        z = F.sigmoid(x + self.lin_hz(dz))
        u = F.tanh(x + self.lin_hh(r * dz))
        dh = (1 - z) * (u - dz)
        #dh = F.sigmoid(self.out(dh))
        dh = F.softmax(self.out(dh), dim=1)
        return dh


class Decoder(nn.Module):
    def __init__(self, rnn_size, hid_size, use_adjoint=False):
        super(Decoder, self).__init__()

        self.use_adjoint = use_adjoint
        self.y_emb = nn.GRU(1, rnn_size, batch_first=True)
        self.fc_y = nn.Linear(rnn_size, hid_size)
        self.odefunc = YODEfunc(hid_size)

    def forward(self, y, zx, t):
        h0 = Variable(torch.zeros(1, y.size(0), args.rnn_size).cuda(0))
        output, emb_y = self.y_emb(y.unsqueeze(2), h0)
        zy = self.fc_y(emb_y).squeeze(0)
        odeint = odeint_adjoint if self.use_adjoint else odeint_normal
        state_t = odeint(self.odefunc, zy, t)
        zt = (state_t * zx).sum(2)

        return zt


class YODEfunc(nn.Module):
    def __init__(self, nhidden):
        super(YODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(nhidden, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, nhidden)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class Model(nn.Module):
    def __init__(self, PXz_emb, ODE_pzx_trans, decode):
        super(Model, self).__init__()

        self.drix_emb = PXz_emb
        self.pzx_trans = ODE_pzx_trans
        self.decoder = decode

        self.optimizer = optim.Adam(self.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 50, 0.5)

    def reparam_sample(self, mean, logvar, N):
        std = torch.exp(0.5 * logvar)
        if N is not None:
            eps = Normal(torch.zeros_like(mean), torch.ones_like(mean)).sample([N, ]).to(device)
        else:
            eps = torch.randn(std.size()).to(mean)
        z = mean + std * eps
        z = torch.split(z, [1] * driving_size, 0)
        return z

    def kl_divergence(self, mu, logvar):
        kl = 0.5*(-1 - logvar + mu.pow(2) + logvar.exp())
        kl = kl.sum(1)
        return kl

    def get_pred(self, tar_y, dri_x, t):
        hx = self.drix_emb(dri_x)
        zx_ = self.pzx_trans(hx, t)

        zt = self.decoder(tar_y, zx_, t)

        return zt.transpose(1, 0)

    def get_data(self, train_loader, tar_mean, tar_std):
        self.train_loader = train_loader
        self.tar_mean = tar_mean
        self.tar_std = tar_std

    def save(self):
        fname = best_log_dir
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        #fname = best_log_dir
        self.load_state_dict(torch.load(fname))

    def _train(self, epoch):
        print('At epoch: {}'.format(epoch))
        self.train()
        loss_list = []
        for i, (inputs, target) in enumerate(self.train_loader):
            inputs = Variable(inputs.view(-1, seq_len, input_size).to(device))
            target = Variable(target.to(device))
            # inputs = inputs.unsqueeze(1)
            train_tar = inputs[:, :, target_col]
            train_dri = inputs[:, :, indep_col[0]:(indep_col[1] + 1)]

            if norm:
                pred = self.get_pred(train_tar, train_dri, train_T)
                pred_y = pred * self.tar_std + self.tar_mean
            else:
                pred_y = self.get_pred(train_tar, train_dri, train_T)

            # Calculate Loss
            loss = criterion(target, pred_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print('step: {}. loss: {}.'.format(i,loss))

            loss_list.append(loss.item())
            torch.cuda.empty_cache()
            if i % 20 == 0:
                print('step_loss:', loss.item())

        total_loss = np.array(loss_list).mean()
        return total_loss

    def test(self, x, y):
        self.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = Variable(torch.Tensor(x).to(device))
                target = Variable(torch.Tensor(y).to(device))
                # w_input = Variable(torch.Tensor(train_input).to(device))
            else:
                inputs = Variable(x)

            train_tar = inputs[:, :, target_col]
            train_dri = inputs[:, :, indep_col[0]:(indep_col[1] + 1)]

            if norm:
                pred = self.get_pred(train_tar, train_dri, test_T)
                pred_y = pred * self.tar_std + self.tar_mean
            else:
                pred_y = self.get_pred(train_tar, train_dri, test_T)

            test_mse = criterion(target, pred_y[:, 1:6])
            test_rmse = torch.sqrt(test_mse)
            test_mae = criterionL1(target, pred_y[:, 1:6])

        return test_rmse, test_mae

    def sample(self, x, y):
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = Variable(torch.Tensor(x).to(device))
                target = Variable(torch.Tensor(y).to(device))
                # w_input = Variable(torch.Tensor(train_input).to(device))
            else:
                inputs = Variable(x)

            train_tar = inputs[:, :, target_col]
            train_dri = inputs[:, :, indep_col[0]:(indep_col[1] + 1)]

            if norm:
                pred = self.get_pred(train_tar, train_dri, test_T)
                pred_y = pred * self.tar_std + self.tar_mean
            else:
                pred_y = self.get_pred(train_tar, train_dri, test_T)

            test_mse = criterion(target, pred_y[:, 1:6])
            test_rmse = torch.sqrt(test_mse)
            test_mae = criterionL1(target, pred_y[:, 1:6])
            test_mse1 = criterion(target[:, [0]], pred_y[:, [1]])
            test_rmse1 = torch.sqrt(test_mse1)
            test_mae1 = criterionL1(target[:, [0]], pred_y[:, [1]])
            test_mse2 = criterion(target[:, [1]], pred_y[:, [2]])
            test_rmse2 = torch.sqrt(test_mse2)
            test_mae2 = criterionL1(target[:, [1]], pred_y[:, [2]])
            test_mse3 = criterion(target[:, [2]], pred_y[:, [3]])
            test_rmse3 = torch.sqrt(test_mse3)
            test_mae3 = criterionL1(target[:, [2]], pred_y[:, [3]])
            test_mse4 = criterion(target[:, [3]], pred_y[:, [4]])
            test_rmse4 = torch.sqrt(test_mse4)
            test_mae4 = criterionL1(target[:, [3]], pred_y[:, [4]])
            test_mse5 = criterion(target[:, [4]], pred_y[:, [5]])
            test_rmse5 = torch.sqrt(test_mse5)
            test_mae5 = criterionL1(target[:, [4]], pred_y[:, [5]])

            return test_mae, test_rmse, test_mae1, test_rmse1, test_mae2, test_rmse2, test_mae3, test_rmse3, test_mae4, \
                   test_rmse4, test_mae5, test_rmse5, pred_y

    def fit(self):
        best_val_mae = float("inf")
        best_val_rmse = float("inf")
        test_mae = 0
        test_rmse = 0
        iter = 0

        for epoch in range(args.num_epochs):
            train_loss = self._train(epoch)
            torch.cuda.empty_cache()
            val_rmse, val_mae = self.test(val_x, val_y)
            # val_rmse, val_mae = eval(test_x, test_y)
            print('Train_Loss: {}. Validation mae: {}. Validation rmse: {}'.format(train_loss, val_mae, val_rmse))
            torch.cuda.empty_cache()

            if val_rmse < best_val_rmse:
                best_val_mae = val_mae
                best_val_rmse = val_rmse
                test_rmse, test_mae = self.test(test_x, test_y)

                self.save()
                print('best_val_mae: {}. best_val_rmse: {}'.format(best_val_mae, best_val_rmse))
                print('test_mae: {}. test_rmse: {}'.format(test_mae, test_rmse))

        print('best_val_mae: {}. best_val_rmse: {}'.format(best_val_mae, best_val_rmse))


def log_test(text_env, errors):
    text_env.write("\n testing error: %s \n\n" % (errors))
    return

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    generator = getGenerator(data_path)
    datagen = generator(data_path, target_col, indep_col, win_size, pre_T,
                        train_share, is_stateful, normalize_pattern)

    # train_x, test_x, train_y, test_y, y_mean, y_std = datagen.with_target3()
    # train_input, test_input, train_target, test_target, y_mean, y_std = datagen.getdata()
    train_x, train_y, val_x, val_y, test_x, test_y, y_mean, y_std = datagen.data_extract_val_arbi2()

    ystd = torch.Tensor([y_std]).to(device)
    ymean = torch.Tensor([y_mean]).to(device)

    # print(" --- Data shapes: ", np.shape(train_x), np.shape(train_y), np.shape(test_x), np.shape(test_y))
    print(" --- Data shapes: ", np.shape(train_x), np.shape(train_y), np.shape(val_x), np.shape(val_y),
          np.shape(test_x), np.shape(test_y))

    batch_size = args.batch_size
    dataset_train = subDataset(train_x, train_y)
    dataset_val = subDataset(val_x, val_y)
    dataset_test = subDataset(test_x, test_y)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)

    input_size = train_x.shape[-1]
    driving_size = input_size - 1
    seq_len = train_x.shape[1]

    for n in range(10):
        print('train number in:', n)
        drix_emb = MultiHeadAttention(args.head, driving_size, args.selfatt_size * args.head)
        pzx_trans = ODE_pzx_trans(driving_size)
        decoder = Decoder(args.rnn_size, driving_size)

        model = Model(drix_emb, pzx_trans, decoder).to(device)

        params = (list(drix_emb.parameters()) + list(pzx_trans.parameters()) + list(decoder.parameters()))
        total_Nparam = sum([pa.nelement() for pa in params])
        print("Number of parameter: %.2f" % (total_Nparam))
        #print(model)
        #print(f"The model has the number of parameters of {count_parameters(model)}")

        model.get_data(train_loader, ymean, ystd)

        # train model
        model.fit()

        with open(log_err_file, "a") as text_file:
            log_test(text_file, [args.lr, batch_size, args.rnn_size, args.selfatt_size, args.head, args.seed, norm])
            # log_test(text_file, [test_rmse, test_mae, seed, norm])

        test_path = os.path.join(args.train_dir, best_log_dir)
        model.load(test_path)
        total_mae, total_rmse, mae_1, rmse_1, mae_2, rmse_2, mae_3, rmse_3, mae_4, rmse_4, mae_5, rmse_5, prediction = model.sample(
            test_x, test_y)

        with open(log_err_file, "a") as text_file:
            log_test(text_file,
                     [total_rmse, total_mae, rmse_1, rmse_2, rmse_3, rmse_4, rmse_5, mae_1, mae_2, mae_3, mae_4, mae_5])

        # if n==1:
        #     saveplot = prediction.cpu().detach().numpy()
        #     np.savetxt(path + 'prediction.csv', saveplot, delimiter=',')
        #     np.savetxt(path + 'target.csv', test_y, delimiter=',')