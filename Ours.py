import os
import torch
import torch.nn
import numpy as np

from torch.nn import Sequential, Linear, ELU, Softplus, functional
from torch_geometric.utils import softmax
from torch_scatter import scatter_max

from utils.reorganizer import relabel_graph, insert_multiple_positions

from utils.graph_transformer_layer import GraphTransformerLayer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def z_to_ptr(z):
    unique_graphs = np.unique(z)
    ptr = [0]  # 开始索引从0开始
    for graph in unique_graphs:
        last_index = np.where(z == graph)[0][-1]  # 当前图的最后一个节点索引
        ptr.append(last_index + 1)  # 索引加1是因为ptr表示每个图的结束边界

    return np.array(ptr)


# 使用小批量进行训练的解释器
class ExplainerBatch(torch.nn.Module):
    def __init__(self, _model, _num_labels, _hidden_size, _use_edge_attr=False):
        super(ExplainerBatch, self).__init__()

        self.model = _model
        self.model = self.model.to(device)
        self.model.eval()

        self.num_labels = _num_labels
        self.hidden_size = _hidden_size
        self.use_edge_attr = _use_edge_attr

        self.temperature = 0.1

        self.edge_action_rep_generator = Sequential(
            Linear(self.hidden_size * (2 + self.use_edge_attr), self.hidden_size * 4),
            ELU(),
            Linear(self.hidden_size * 4, self.hidden_size * 2),
            ELU(),
            Linear(self.hidden_size * 2, self.hidden_size)
        ).to(device)

        self.edge_action_prob_generator = self.build_edge_action_prob_generator()

    def build_edge_action_prob_generator(self):
        edge_action_prob_generator = Sequential(Linear(self.hidden_size, self.hidden_size),
                                                ELU(),
                                                Linear(self.hidden_size, self.num_labels),
                                                ELU()
                                                ).to(device)
        return edge_action_prob_generator

    def predict(self, cand_action_reps, target_y, cand_action_batch):
        action_probs = self.edge_action_prob_generator(cand_action_reps)  # m*n;m候选边数量，n分类标签数量
        action_probs = action_probs.gather(1, target_y.view(-1, 1))  # 找到graph对应的实际标签，取该标签对应的可能
        action_probs = action_probs.reshape(-1)  # 重整为m*1的张量

        action_probs = softmax(action_probs, cand_action_batch)
        return action_probs

    def forward(self, graph, state, train_flag=False):
        ocp_edge_index = graph.edge_index.T[state].T
        ocp_edge_attr = graph.edge_attr[state]

        cand_edge_index = graph.edge_index.T[~state].T
        cand_edge_attr = graph.edge_attr[state]

        cand_node_reps_0 = self.model.get_node_reps(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        cand_node_reps_1 = self.model.get_node_reps(graph.x, ocp_edge_index, ocp_edge_attr, graph.batch)
        cand_node_reps = cand_node_reps_0 - cand_node_reps_1

        # cand_edge_index:2*m; torch.cat(m*d, m*d)->m*2d; 其中m为候选边个数；d为节点特征数
        cand_action_reps = torch.cat([cand_node_reps[cand_edge_index[0]],
                                      cand_node_reps[cand_edge_index[1]]], dim=1).to(device)
        cand_action_reps = self.edge_action_rep_generator(cand_action_reps)

        cand_action_batch = graph.batch[cand_edge_index[0]]
        cand_y_batch = graph.y[cand_action_batch]

        if self.use_edge_attr:
            cand_edge_reps = self.model.edge_emb(cand_edge_attr)
            cand_action_reps = torch.cat([cand_action_reps, cand_edge_reps], dim=1)

        cand_action_probs = self.predict(cand_action_reps, cand_y_batch, cand_action_batch)

        # 找到每张图中候选边可能性最高的“可能”和“边”
        added_action_probs, added_actions = scatter_max(cand_action_probs, cand_action_batch)

        if train_flag:
            rand_action_probs = torch.rand(cand_action_probs.size()).to(device)
            # '_'表示抛弃了最大值rand_action_probs
            _, rand_actions = scatter_max(rand_action_probs, cand_action_batch)
            return cand_action_probs, cand_action_probs[rand_actions], rand_actions

        return cand_action_probs, added_action_probs, added_actions

    def get_optimizer(self, lr=0.0001, weight_decay=1e-5, scope='all'):
        if scope in ['all']:
            params = self.parameters()
        else:
            params = list(self.edge_action_rep_generator.parameters()) + \
                     list(self.edge_action_prob_generator.parameters())

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return optimizer

    def load_policy_net(self, name='policy.pt', path=None):
        if not path:
            path = os.path.join(os.path.dirname(__file__), '../params', name)
        self.load_state_dict(torch.load(path))

    def save_policy_net(self, name='policy.pt', path=None):
        if not path:
            path = os.path.join(os.path.dirname(__file__), '../params', name)
        torch.save(self.state_dict(), path)


class ExplainerBatchStar(ExplainerBatch):
    def __init__(self, _model, _num_labels, _hidden_size, _use_edge_attr=False):
        super(ExplainerBatchStar, self).__init__(_model, _num_labels, _hidden_size, _use_edge_attr=False)

        self.edge_action_prob_generator = self.build_edge_action_prob_generator()

    def build_edge_action_prob_generator(self):
        edge_action_prob_generator = torch.nn.ModuleList()
        for i in range(self.num_labels):
            explainer_i = Sequential(Linear(self.hidden_size * (2 + self.use_edge_attr), self.hidden_size * 2),
                                     ELU(),
                                     Linear(self.hidden_size * 2, self.hidden_size),
                                     ELU(),
                                     Linear(self.hidden_size, 1)
                                     ).to(device)
            edge_action_prob_generator.append(explainer_i)

        return edge_action_prob_generator

    def predict_star(self, graph_rep, subgraph_rep, cand_action_reps, target_y, cand_action_batch):
        action_graph_reps = graph_rep - subgraph_rep
        action_graph_reps = action_graph_reps[cand_action_batch]
        action_graph_reps = torch.cat([cand_action_reps, action_graph_reps], dim=1)

        action_probs = []
        for explainer_i in self.edge_action_prob_generator:
            i_action_probs = explainer_i(action_graph_reps)
            action_probs.append(i_action_probs)
        action_probs = torch.cat(action_probs, dim=1)

        action_probs = action_probs.gather(1, target_y.view(-1, 1))
        action_probs = action_probs.reshape(-1)

        return action_probs

    def forward(self, graph, state, train_flag=False):
        graph_rep = self.model.get_graph_reps(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

        # 如果state全为False，即解释子图为空
        if torch.where(state)[0].numel() == 0:
            subgraph_rep = torch.zeros(graph_rep.size()).to(device)
        else:
            subgraph = relabel_graph(graph, state)
            subgraph_rep = self.model.get_graph_reps(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)

        cand_edge_index = graph.edge_index.T[~state].T
        cand_edge_attr = graph.edge_attr[~state]
        cand_node_reps = self.model.get_node_reps(graph.x, cand_edge_index, cand_edge_attr, graph.batch)

        if self.use_edge_attr:
            cand_edge_reps = self.model.edge_emb(cand_edge_attr)
            cand_action_reps = torch.cat([cand_node_reps[cand_edge_index[0]],
                                          cand_node_reps[cand_edge_index[1]],
                                          cand_edge_reps], dim=1).to(device)
        else:
            cand_action_reps = torch.cat([cand_node_reps[cand_edge_index[0]],
                                          cand_node_reps[cand_edge_index[1]]], dim=1).to(device)

        cand_action_reps = self.edge_action_rep_generator(cand_action_reps)

        cand_action_batch = graph.batch[cand_edge_index[0]]
        cand_y_batch = graph.y[cand_action_batch]

        # 防止batches超出候选actions之外
        unique_batch, cand_action_batch = torch.unique(cand_action_batch, return_inverse=True)

        cand_action_probs = self.predict_star(graph_rep, subgraph_rep,
                                              cand_action_reps, cand_y_batch, cand_action_batch)

        assert len(cand_action_probs) == sum(~state)

        # 根据cand_action_batch中的索引；将cand_action_probs中同一索引的最大值提出赋给added_action_probs，
        # added_actions记录对应added_action_probs中元素在cand_action_probs中的位置
        added_action_probs, added_actions = scatter_max(cand_action_probs, cand_action_batch)

        if train_flag:
            rand_action_probs = torch.rand(cand_action_probs.size()).to(device)
            _, rand_actions = scatter_max(rand_action_probs, cand_action_batch)

            return cand_action_probs, cand_action_probs[rand_actions], rand_actions, unique_batch

        return cand_action_probs, added_action_probs, added_actions, unique_batch


class ExplainerBatchGraphTransform(ExplainerBatchStar):
    def __init__(self, _model, _num_labels, _hidden_size, _use_edge_attr=False):
        super(ExplainerBatchStar, self).__init__(_model, _num_labels, _hidden_size, _use_edge_attr=False)

        # self.subgraph_reps_generator = self.build_subgraph_reps_generator()
        self.new_feature_generator = self.get_new_feature()
        self.merge = Linear(2 * self.hidden_size, self.hidden_size)
        # self.melt = Linear(2 * self.hidden_size, 1)

    # def build_subgraph_reps_generator(self):
    #     subgraph_reps_generator = Sequential(Linear(self.hidden_size, self.hidden_size * 2), ELU(),
    #                                          Linear(self.hidden_size * 2, self.hidden_size), ELU(),
    #                                          Linear(self.hidden_size, 1),
    #                                          Softplus()).to(device)
    #     return subgraph_reps_generator

    def get_new_feature(self):
        build_new_feature_generator = GraphTransformerLayer(in_dim=self.hidden_size, hidden_dim=self.hidden_size,
                                                            edge_dim=None, n_heads=4, dropout=0.1,
                                                            use_edges=self.use_edge_attr, batch_norm=True)
        return build_new_feature_generator

    def get_optimizer(self, lr=0.1, weight_decay=1e-3, scope='all'):
        if scope in ['all']:
            params = self.parameters()
        else:
            params = list(self.edge_action_rep_generator.parameters()) + \
                     list(self.edge_action_prob_generator.parameters()) + \
                     list(self.new_feature_generator.parameters()) + \
                     list(self.merge.parameters())

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return optimizer

    def forward_pro(self, graph, state, train_flag=False):
        graph_rep = self.model.get_graph_reps(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

        # 如果state全为False，即解释子图为空
        if torch.where(state)[0].numel() == 0:
            subgraph_rep = torch.zeros(graph_rep.size()).to(device)
        else:
            subgraph = relabel_graph(graph, state)
            subgraph_rep = self.model.get_graph_reps(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)

            # subgraph_pred = self.model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)
            # subgraph_acc = (graph.y == subgraph_pred.argmax(dim=1)).sum().item()
            # subgraph_acc = subgraph_acc / len(graph)

        cand_edge_index = graph.edge_index.T[~state].T
        cand_edge_attr = graph.edge_attr[~state]
        cand_node_reps_1 = self.model.get_node_reps(graph.x, cand_edge_index, cand_edge_attr, graph.batch)

        cand_node_reps = self.new_feature_generator(cand_node_reps_1, graph.edge_index, self.use_edge_attr)[0]
        cand_node_reps = self.merge(torch.cat((cand_node_reps, cand_node_reps_1), dim=1).to(device))
        # cand_node_reps = cand_node_reps_1

        if self.use_edge_attr:
            cand_edge_reps = self.model.edge_emb(cand_edge_attr)
            cand_action_reps = torch.cat([cand_node_reps[cand_edge_index[0]],
                                          cand_node_reps[cand_edge_index[1]],
                                          cand_edge_reps], dim=1).to(device)
        else:
            cand_action_reps = torch.cat([cand_node_reps[cand_edge_index[0]],
                                          cand_node_reps[cand_edge_index[1]]], dim=1).to(device)

        # ptr = batch_to_ptr(graph.batch)

        cand_action_reps = self.edge_action_rep_generator(cand_action_reps) + 1

        cand_action_batch = graph.batch[cand_edge_index[0]]
        cand_y_batch = graph.y[cand_action_batch]

        # 防止batches超出候选actions之外
        unique_batch, cand_action_batch = torch.unique(cand_action_batch, return_inverse=True)

        cand_action_probs = self.predict_star(graph_rep, subgraph_rep, cand_action_reps, cand_y_batch, cand_action_batch)
        # cand_action_probs = self.melt(cand_action_reps).to(device)
        # cand_action_probs = cand_action_probs.squeeze(1)

        # cand_action_probs = functional.normalize(cand_action_probs, dim=0)

        assert len(cand_action_probs) == sum(~state)
        assert len(cand_action_probs) == len(cand_action_batch)

        added_action_probs, added_actions = scatter_max(cand_action_probs, cand_action_batch)

        flag = []
        if len(unique_batch) < len(graph.y):
            long = len(graph.y)
            full_set = set(range(long))
            batch_set = set(unique_batch.tolist())
            missing_index = sorted(list(full_set - batch_set))

            flag = missing_index
            unique_batch = torch.tensor(list(full_set)).to(device)
            added_actions = insert_multiple_positions(added_actions, missing_index, -1)
            added_action_probs = insert_multiple_positions(added_action_probs, missing_index, 0)

        # if torch.where(state)[0].numel() == 0:
        #     added_action_probs, added_actions = scatter_max(cand_action_probs, cand_action_batch)
        # else:
        #     subgraph_probs = self.subgraph_reps_generator(subgraph_rep)
        #     subgraph_probs = subgraph_probs.view(-1)
        #     subgraph_probs = functional.normalize(subgraph_probs, dim=0)
        #     temp_subgraph_probs = subgraph_probs[cand_action_batch]
            # temp_subgraph_probs = torch.zeros_like(cand_action_batch, dtype=torch.float)
            # temp_subgraph_probs.scatter_(0, cand_action_batch, subgraph_probs[cand_action_batch])
            # temp_action_probs = (cand_action_probs / temp_subgraph_probs) - 1
            # temp_action_probs = (cand_action_probs - temp_pre_probs) / (torch.abs(temp_pre_probs) + 1e-10)
            # temp_action_probs = (cand_action_probs - subgraph_acc) / (subgraph_acc + 1e-10)
            # added_action_probs, added_actions = scatter_max(temp_action_probs, cand_action_batch)

            # temp_action_probs = []
            # for batch in unique_batch:
            #     batch_mask = cand_action_batch == batch
            #
            #     batch_probs = cand_action_probs[batch_mask]
            #     temp_action_prob = (batch_probs - pre_probs[batch]) / pre_probs[batch]
            #
            #     temp_action_probs.append(temp_action_prob)
            #
            # temp_action_probs = [t for t in temp_action_probs if t.numel() > 0]
            # temp_action_probs = torch.cat(temp_action_probs)
            # assert len(temp_action_probs) == len(cand_action_batch)
            # added_action_probs, added_actions = scatter_max(temp_action_probs, cand_action_batch)
            # added_actions = added_actions - 1

        if train_flag:
            rand_action_probs = torch.rand(cand_action_probs.size()).to(device)
            rand_action_probs, rand_actions = scatter_max(rand_action_probs, cand_action_batch)

            if len(rand_actions) < len(graph.y):
                rand_actions = insert_multiple_positions(rand_actions, flag, -1)
                rand_action_probs = insert_multiple_positions(rand_action_probs, flag, -1)
            return cand_action_probs, rand_action_probs, rand_actions, unique_batch

        return cand_action_probs, added_action_probs, added_actions, unique_batch




