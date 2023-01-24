import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math
import random
from action_utils import select_action, translate_action
from gnn_layers import GraphAttention


class MAGIC(nn.Module):
    """
    The communication protocol of Multi-Agent Graph AttentIon Communication (MAGIC)
    """

    def __init__(self, args):
        super(MAGIC, self).__init__()
        """
        Initialization method for the MAGIC communication protocol (2 rounds of communication)

        Arguements:
            args (Namespace): Parse arguments
        """

        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size

        dropout = 0
        negative_slope = 0.2

        # initialize sub-processors
        self.sub_processor1 = GraphAttention(args.hid_size, args.gat_hid_size, dropout=dropout,
                                             negative_slope=negative_slope, num_heads=args.gat_num_heads,
                                             self_loop_type=args.self_loop_type1, average=False,
                                             normalize=args.first_gat_normalize)
        self.sub_processor2 = GraphAttention(args.gat_hid_size * args.gat_num_heads, args.hid_size, dropout=dropout,
                                             negative_slope=negative_slope, num_heads=args.gat_num_heads_out,
                                             self_loop_type=args.self_loop_type2, average=True,
                                             normalize=args.second_gat_normalize)
        # initialize the gat encoder for the Scheduler
        if args.use_gat_encoder:
            self.gat_encoder = GraphAttention(args.hid_size, args.gat_encoder_out_size, dropout=dropout,
                                              negative_slope=negative_slope, num_heads=args.ge_num_heads,
                                              self_loop_type=1, average=True, normalize=args.gat_encoder_normalize)

        self.obs_encoder = nn.Linear(args.obs_size, args.hid_size)

        self.init_hidden(args.batch_size)
        self.lstm_cell = nn.LSTMCell(args.hid_size, args.hid_size)

        if args.message_encoder:
            self.message_encoder = nn.Linear(args.hid_size, args.hid_size)
        if args.message_decoder:
            self.message_decoder = nn.Linear(args.hid_size, args.hid_size)

        # initialize weights as 0
        if args.comm_init == 'zeros':
            if args.message_encoder:
                self.message_encoder.weight.data.zero_()
            if args.message_decoder:
                self.message_decoder.weight.data.zero_()
            if not args.first_graph_complete:
                self.sub_scheduler_mlp1.apply(self.init_linear)
            if args.learn_second_graph and not args.second_graph_complete:
                self.sub_scheduler_mlp2.apply(self.init_linear)

        # initialize the action head (in practice, one action head is used)
        self.action_heads = nn.ModuleList([nn.Linear(2 * args.hid_size, o)
                                           for o in args.naction_heads])
        # initialize the value head
        self.value_head = nn.Linear(2 * self.hid_size, 1)

        self.window = 3
        self.message_size = self.hid_size
        self.history_index_window = {t: torch.zeros(self.nagents, self.nagents) for t in range(-self.window, 0)}
        self.history_message_window = [torch.zeros(1, self.nagents * self.message_size) for _ in range(-self.window, 0)]
        self.history_window = {t: [] for t in range(-self.window, 0)}
        
        self.fe = lambda epoch_idx: math.exp(-1. * epoch_idx / 20)

    def init_windows(self):
        self.history_index_window = {t: torch.zeros(self.nagents, self.nagents) for t in range(-self.window, 0)}
        self.history_message_window = [torch.zeros(1, self.nagents * self.message_size) for _ in range(-self.window, 0)]
        self.history_window = {t: [] for t in range(-self.window, 0)}

    def forward(self, x, memodel, t, epoch, info={}):
        """
        Forward function of MAGIC (two rounds of communication)

        Arguments:
            x (list): a list for the input of the communication protocol [observations, (previous hidden states, previous cell states)]
            observations (tensor): the observations for all agents [1 (batch_size) * n * obs_size]
            previous hidden/cell states (tensor): the hidden/cell states from the previous time steps [n * hid_size]

        Returns:
            action_out (list): a list of tensors of size [1 (batch_size) * n * num_actions] that represent output policy distributions
            value_head (tensor): estimated values [n * 1]
            next hidden/cell states (tensor): next hidden/cell states [n * hid_size]
        """

        # n: number of agents
        n = self.nagents
        obs, extras = x
        # encoded_obs: [1 (batch_size) * n * hid_size]
        encoded_obs = self.obs_encoder(obs)
        hidden_state, cell_state = extras
        batch_size = encoded_obs.size()[0]

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        # if self.args.comm_mask_zero == True, block the communiction (can also comment out the protocol to make training faster)
        if self.args.comm_mask_zero:
            agent_mask *= torch.zeros(n, 1)

        hidden_state, cell_state = self.lstm_cell(encoded_obs.squeeze(), (hidden_state, cell_state))
        if epoch > 300:
            hidden_state, cell_state = hidden_state.detach(), cell_state.detach()

        # comm: [n * hid_size]
        comm = hidden_state
        if self.args.message_encoder:
            comm = self.message_encoder(comm)

        message_curr = comm.clone()

        adj_complete = self.get_complete_graph(agent_mask)

        # do estimation
        if t == 0:
            self.last_message = torch.zeros(self.nagents, self.nagents, self.message_size)

        me_last = self.last_message.view(self.nagents, self.nagents, self.message_size)
        ac_est = self.get_est_action(me_last, self.nagents, self.nagents)

        ac_est = ac_est.view(self.nagents, -1)
        ac_est = ac_est.repeat(self.nagents, 1, 1)
        ma0 = torch.cat((ac_est, me_last), dim=-1).view(self.nagents, -1)
        cm_ = memodel(ma0)
        cm_ = cm_.view(self.nagents, self.nagents, self.message_size)
        
        e = self.fe(epoch)
        rand = random.random()

        # get communication objectives
        if t == 0 or rand <= e:
            # self.curr_message = comm.repeat(self.nagents, 1)
            comm_adj = adj_complete
        else:
            comm_adj = self.get_comm_obj(cm_, message_curr)

        comm_adj = comm_adj.detach()

        # receive from other agents
        cm = comm.repeat(self.nagents, 1, 1)
        cm = comm_adj.unsqueeze(-1) * cm

        # use the message estimation for the other agents
        cm_ = (torch.ones(self.nagents, self.nagents) - comm_adj).unsqueeze(-1) * cm_

        # combine these messages
        self.curr_message = cm + cm_

        comm_curr = self.curr_message.clone()

        comm_curr = comm_curr.view(self.nagents, self.nagents, self.hid_size)
        comm_curr = F.elu(self.sub_processor1(comm_curr, adj_complete))
        comm_curr = self.sub_processor2(comm_curr, adj_complete)
        
        # self.last_cm = comm_curr.clone().detach()

        comm_curr_self_message = []
        for i in range(self.nagents):
            curr_i = comm_curr[i, i, :].unsqueeze(0)
            comm_curr_self_message.append(curr_i)
        comm = torch.cat(comm_curr_self_message, dim=0)

        comm = comm * agent_mask
        comm_ori = comm.clone()

        comm = comm * agent_mask

        if self.args.message_decoder:
            comm = self.message_decoder(comm)

        value_head = self.value_head(torch.cat((hidden_state, comm), dim=-1))
        h = hidden_state.view(batch_size, n, self.hid_size)
        c = comm.view(batch_size, n, self.hid_size)

        action_out = [F.log_softmax(action_head(torch.cat((h, c), dim=-1)), dim=-1) for action_head in
                      self.action_heads]

        self.last_message = self.curr_message.clone().detach()
        action_model = action_out[0].detach()
        action_model = action_model.repeat(self.nagents, 1, 1).view(self.nagents, -1)
        m_a_curr = torch.cat((action_model, message_curr), dim=1)
        m_a_curr = m_a_curr.view(-1)
        message_curr = message_curr.view(-1)

        return action_out, value_head, (hidden_state.clone(), cell_state.clone()), m_a_curr.detach(), message_curr.detach(), comm_adj

    def get_est_action(self, comm, batch_size, n):
        h = comm.clone().view(batch_size, n, self.hid_size)

        adj1 = torch.ones(self.nagents, self.nagents)
        comm = comm.view(self.nagents, self.nagents, self.hid_size)
        comm = F.elu(self.sub_processor1(comm, adj1))
        comm = self.sub_processor2(comm, adj1)

        c = comm.view(batch_size, n, self.hid_size)
        action_est = [F.log_softmax(action_head(torch.cat((h, c), dim=-1)), dim=-1) for action_head in
                      self.action_heads]
        return action_est[0].detach()

    def get_agent_mask(self, batch_size, info):
        """
        Function to generate agent mask to mask out inactive agents (only effective in Traffic Junction)

        Returns:
            num_agents_alive (int): number of active agents
            agent_mask (tensor): [n, 1]
        """

        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(n, 1).clone()

        return num_agents_alive, agent_mask

    def init_linear(self, m):
        """
        Function to initialize the parameters in nn.Linear as o
        """
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)

    def init_hidden(self, batch_size):
        """
        Function to initialize the hidden states and cell states
        """
        return tuple((torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                      torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

    def sub_scheduler(self, sub_scheduler_mlp, hidden_state, agent_mask, directed=True):
        """
        Function to perform a sub-scheduler

        Arguments:
            sub_scheduler_mlp (nn.Sequential): the MLP layers in a sub-scheduler
            hidden_state (tensor): the encoded messages input to the sub-scheduler [n * hid_size]
            agent_mask (tensor): [n * 1]
            directed (bool): decide if generate directed graphs

        Return:
            adj (tensor): a adjacency matrix which is the communication graph [n * n]
        """

        # hidden_state: [n * hid_size]
        n = self.args.nagents
        hid_size = hidden_state.size(-1)
        # hard_attn_input: [n * n * (2*hid_size)]
        hard_attn_input = torch.cat([hidden_state.repeat(1, n).view(n * n, -1), hidden_state.repeat(n, 1)], dim=1).view(
            n, -1, 2 * hid_size)
        # hard_attn_output: [n * n * 2]
        if directed:
            hard_attn_output = F.gumbel_softmax(sub_scheduler_mlp(hard_attn_input), hard=True)
        else:
            hard_attn_output = F.gumbel_softmax(
                0.5 * sub_scheduler_mlp(hard_attn_input) + 0.5 * sub_scheduler_mlp(hard_attn_input.permute(1, 0, 2)),
                hard=True)
        # hard_attn_output: [n * n * 1]
        hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)
        # agent_mask and agent_mask_transpose: [n * n]
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        # adj: [n * n]
        adj = hard_attn_output.squeeze() * agent_mask * agent_mask_transpose

        return adj

    def get_comm_obj(self, cm, message_curr):
        cm_i_list = []
        for i in range(self.nagents):
            curr_i = cm[i, i, :].unsqueeze(0)
            cm_i_list.append(curr_i)
        message_curr_est = torch.cat(cm_i_list, dim=0)
        differnece = message_curr_est - message_curr
        d = torch.pow(differnece, 2)
        sum = torch.mean(d, -1)
        ad = torch.where(sum >= 0.2, 1, 0)
        adj = ad.repeat(self.nagents, 1)
        return adj


    def get_complete_graph(self, agent_mask):
        """
        Function to generate a complete graph, and mask it with agent_mask
        """
        n = self.args.nagents
        adj = torch.ones(n, n)
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        adj = adj * agent_mask * agent_mask_transpose

        return adj
