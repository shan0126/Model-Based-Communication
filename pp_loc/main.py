import sys
import time
import signal
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import visdom
import data
from magic import MAGIC
from utils import *
from action_utils import parse_action_args
from trainer import Trainer
from multiprocessing import Process
import gym
from message_model import *

gym.logger.set_level(40)

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='Multi-Agent Graph Attention Communication')

# training
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=16,
                    help='How many processes to run')

# model
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--directed', action='store_true', default=False,
                    help='whether the communication graph is directed')
parser.add_argument('--self_loop_type1', default=2, type=int,
                    help='self loop type in the first gat layer (0: no self loop, 1: with self loop, 2: decided by hard attn mechanism)')
parser.add_argument('--self_loop_type2', default=2, type=int,
                    help='self loop type in the second gat layer (0: no self loop, 1: with self loop, 2: decided by hard attn mechanism)')
parser.add_argument('--gat_num_heads', default=1, type=int,
                    help='number of heads in gat layers except the last one')
parser.add_argument('--gat_num_heads_out', default=1, type=int,
                    help='number of heads in output gat layer')
parser.add_argument('--gat_hid_size', default=64, type=int,
                    help='hidden size of one head in gat')
parser.add_argument('--ge_num_heads', default=4, type=int,
                    help='number of heads in the gat encoder')
parser.add_argument('--first_gat_normalize', action='store_true', default=False,
                    help='whether normalize the coefficients in the first gat layer of the message processor')
parser.add_argument('--second_gat_normalize', action='store_true', default=False,
                    help='whether normilize the coefficients in the second gat layer of the message proccessor')
parser.add_argument('--gat_encoder_normalize', action='store_true', default=False,
                    help='whether normilize the coefficients in the gat encoder (they have been normalized if the input graph is complete)')
parser.add_argument('--use_gat_encoder', action='store_true', default=False,
                    help='whether use the gat encoder before learning the first graph')
parser.add_argument('--gat_encoder_out_size', default=64, type=int,
                    help='hidden size of output of the gat encoder')
parser.add_argument('--first_graph_complete', action='store_true', default=False,
                    help='whether the first communication graph is set to a complete graph')
parser.add_argument('--second_graph_complete', action='store_true', default=False,
                    help='whether the second communication graph is set to a complete graph')
parser.add_argument('--learn_second_graph', action='store_true', default=False,
                    help='whether learn a new communication graph at the second round of communication')
parser.add_argument('--message_encoder', action='store_true', default=False,
                    help='whether use the message encoder')
parser.add_argument('--message_decoder', action='store_true', default=False,
                    help='whether use the message decoder')
parser.add_argument('--nagents', type=int, default=1,
                    help="number of agents")
parser.add_argument('--mean_ratio', default=0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--detach_gap', default=10000, type=int,
                    help='detach hidden state and cell state for rnns at this interval')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="whether block the communication")

# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed')
parser.add_argument('--n_experiments', '-e', type=int, default=10)
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coefficient for value loss term')

# environment
parser.add_argument('--env_name', default="predator_prey",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')

# other
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--plot_port', default='8097', type=str,
                    help='plot port')
parser.add_argument('--save', action="store_true", default=False,
                    help='save the model after training')
parser.add_argument('--save_every', default=0, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--display', action="store_true", default=False,
                    help='display environment state')
parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")


init_args_for_env(parser)
args = parser.parse_args()

args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")


def multi_process_run(args, seed):
    print('start program with random seed {}'.format(seed))
    env = data.init(args.env_name, args, False)
    args.obs_size = env.observation_dim
    args.num_actions = env.num_actions
    if not isinstance(args.num_actions, (list, tuple)):  # single action case
        args.num_actions = [args.num_actions]
    args.dim_actions = env.dim_actions

    parse_action_args(args)

    torch.manual_seed(seed)

    policy_net = MAGIC(args)
    model_net = MessageModel(args)

    disp_trainer = Trainer(args, policy_net, model_net, data.init(args.env_name, args, False))
    disp_trainer.display = True

    def disp():
        x = disp_trainer.get_episode()

    trainer = Trainer(args, policy_net, model_net, data.init(args.env_name, args))

    log = dict()
    log['epoch'] = LogField(list(), False, None, None)
    log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['reward_log_2'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['reward_log_05'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['reward_log_1'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['reward_log_5'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['channel_sum'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
    log['start_model_loss'] = LogField(list(), True, 'epoch', 'num_steps')
    log['end_model_loss'] = LogField(list(), True, 'epoch', 'num_steps')
    log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
    log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
    log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
    log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')

    if args.plot:
        vis = visdom.Visdom(env=args.plot_env, port=args.plot_port)

    model_dir = Path('./saved') / args.env_name

    curr_run = 'run_seed'+ str(seed)
    run_dir = model_dir / curr_run

    run(args.num_epochs, run_dir, trainer, log, seed, policy_net, model_net)


print(args)


# share parameters among threads, but not gradients
# for p in policy_net.parameters():
#     p.data.share_memory_()


def run(num_epochs, run_dir, trainer, log, seed, policy_net, model_net):
    
    num_episodes = 0
    if args.save:
        os.makedirs(run_dir)
    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        stat = dict()
        strs = []
        for n in range(args.epoch_size*16):
            if n == args.epoch_size - 1 and args.display:
                trainer.display = True
            s = trainer.train_batch(ep)
            if n%16 == 0:
                strs.append('batch: {}'.format( n/16))
            merge_stat(s, stat)
            trainer.display = False

        epoch_time = time.time() - epoch_begin_time
        epoch = len(log['epoch'].data) + 1
        num_episodes += stat['num_episodes']
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        np.set_printoptions(precision=2)
        
        
        strs.append('Epoch {}'.format(epoch))
        strs.append('Episode: {}'.format(num_episodes))
        strs.append('Reward: {}'.format(stat['reward']))
        strs.append('reward_log_2: {}'.format(stat['reward_log_2']))
        strs.append('reward_log_05: {}'.format(stat['reward_log_05']))
        strs.append('reward_log_1: {}'.format(stat['reward_log_1']))
        strs.append('reward_log_5: {}'.format(stat['reward_log_5']))
        strs.append('sloss: {}'.format(stat['start_model_loss']))
        strs.append('eloss: {}'.format(stat['end_model_loss']))
        strs.append('Channels: {}'.format(stat['channel_sum']))
        strs.append('Time: {:.2f}s'.format(epoch_time))
        
        
        if 'enemy_reward' in stat.keys():
            strs.append('Enemy-Reward: {}'.format(stat['enemy_reward']))
        if 'add_rate' in stat.keys():
            strs.append('Add-Rate: {:.2f}'.format(stat['add_rate']))
        if 'success' in stat.keys():
            strs.append('Success: {:.4f}'.format(stat['success']))
        if 'steps_taken' in stat.keys():
            strs.append('Steps-Taken: {:.2f}'.format(stat['steps_taken']))
        if 'comm_action' in stat.keys():
            strs.append('Comm-Action: {}'.format(stat['comm_action']))
        if 'enemy_comm' in stat.keys():
            strs.append('Enemy-Comm: {}'.format(stat['enemy_comm']))
            

        for str_ in strs:
            print(str_)

        # if args.plot:
        #     for k, v in log.items():
        #         if v.plot and len(v.data) > 0:
        #             vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
        #             win=k, opts=dict(xlabel=v.x_axis, ylabel=k))
    
        if args.save_every and ep and args.save and (ep+1) % args.save_every == 0:
            save(False, policy_net, model_net, log, trainer, run_dir, epoch=ep+1)

        if args.save:
            save(True, policy_net, model_net, log, trainer, run_dir)




def save(final, policy_net, model_net, log, trainer, run_dir, epoch=0):
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    d['model'] = model_net.state_dict()
    if final:
        torch.save(d, run_dir / 'model.pt')
    else:
        torch.save(d, run_dir / ('model_ep%i.pt' %(epoch)))


def load(path, policy_net, model_net, log, trainer):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    model_net.load_state_dict(d['model'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])

def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if args.load != '':
    load(args.load)

processes = []
for e in range(args.n_experiments):
    seed = args.seed + e
    print('Running experiment with seed %d'%seed)
    def train_func():
        multi_process_run(args, seed)
    p = Process(target=train_func, args=tuple())
    p.start()
    processes.append(p)

for p in processes:
    p.join()


if sys.flags.interactive == 0 and args.nprocesses > 1:
    import os
    os._exit(0)



