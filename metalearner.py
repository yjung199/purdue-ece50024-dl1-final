from __future__ import division, print_function, absolute_import

import torch
import torch.nn as nn

class MetaLSTMCell(nn.Module):
    def __init__(self, input_size=20, hidden_size=1, n_learner_params=0):
        super().__init__()
        # Abbr. from https://github.com/twitter-research/meta-learning-lstm
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_learner_params = n_learner_params
        self.WF = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        self.WI = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        self.cI = nn.Parameter(torch.Tensor(n_learner_params, 1))
        self.bI = nn.Parameter(torch.Tensor(1, hidden_size))
        self.bF = nn.Parameter(torch.Tensor(1, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            nn.init.uniform_(weight, -0.01, 0.01)

        nn.init.uniform_(self.bF, 4, 6)
        nn.init.uniform_(self.bI, -5, -4)

    def init_cI(self, flat_params):
        self.cI.data.copy_(flat_params.unsqueeze(1))


    def forward(self, inputs, hx=None):
        x_all, grad = inputs
        batch, _ = x_all.size()

        if hx is None:
            f_prev = torch.zeros((batch, self.hidden_size), device=self.WF.device)
            i_prev = torch.zeros((batch, self.hidden_size), device=self.WI.device)
            c_prev = self.cI
        else:
            f_prev, i_prev, c_prev = hx

        f_next = torch.mm(torch.cat((x_all, c_prev, f_prev), dim=1), self.WF) + self.bF.expand_as(f_prev)
        i_next = torch.mm(torch.cat((x_all, c_prev, i_prev), dim=1), self.WI) + self.bI.expand_as(i_prev)
        c_next = torch.sigmoid(f_next).mul(c_prev) - torch.sigmoid(i_next).mul(grad)

        return c_next, [f_next, i_next, c_next]
    
     
    def extra_repr(self):
        return f"{self.input_size}, {self.hidden_size}, {self.n_learner_params}"

# Path: metalearner.py
class MetaLearner(nn.Module):

    def __init__(self, input_size=4, hidden_size=20, n_learner_params=0):
        super().__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.metalstm = MetaLSTMCell(hidden_size, 1, n_learner_params)

    def forward(self, inputs, hs=None):
        loss, grad_prep, grad = inputs
        loss = loss.expand_as(grad_prep)
        inputs = torch.cat((loss, grad_prep), dim=1)
        if hs is None:
            hs = [None, None]
        lstmhx, lstmcx = self.lstm(inputs, hs[0])
        flat_learner_unsqzd, metalstm_hs = self.metalstm([lstmhx, grad], hs[1])
        return flat_learner_unsqzd.squeeze(), [(lstmhx, lstmcx), tuple(metalstm_hs)]