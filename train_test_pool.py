import numpy as np
import torch
import torch.nn.functional as functional
from torch.optim import lr_scheduler

from tqdm import tqdm
from torch_scatter import scatter_max

from utils.reorganizer import relabel_graph, insert_multiple_positions
from utils.SpectralCluster import compute_batch_laplacian, spectral_cluster, community_cluster

EPS = 8e-3
# EPS = 1e-15
k = 0.5
# k = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parameters_changed(model, prev_params):
    for param, prev_param in zip(model.parameters(),
                                 prev_params):
        if not torch.equal(param, prev_param):
            return True
    return False


# 在不同的解释子图的大小限制下，计算对应的预测准确率
def test_policy_all_with_gnd(explainer, model, test_loader, top_n=None):
    explainer.eval()
    model.eval()

    # 限制解释器预测的解释子图的大小
    top_k_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # 存储对应限制下的准确性计数表
    acc_count_list = np.zeros(len(top_k_ratio_list))

    precision_top_n_count = 0
    recall_top_n_count = 0

    num_graph = 0
    with torch.no_grad():
        for graph in tqdm(iter(test_loader)):
            graph = graph.to(device)
            num_graph += len(graph)
            max_budget = int(graph.num_edges / num_graph)
            state = torch.zeros(graph.num_edges, dtype=torch.bool)

            check_budget_list = [max(int(_top_k * max_budget), 1) for _top_k in top_k_ratio_list]
            valid_budget = max(int(0.9 * max_budget), 1)

            # pre_probs = torch.empty(0, 0)
            for budget in range(valid_budget):
                # 初始化时，解释子图为空，所有的边均为侯选边
                available_actions = state[~state].clone()

                _, make_action_probs, make_action_id, _ = explainer.forward_pro(graph=graph, state=state,
                                                                                train_flag=False)

                # pre_probs = make_action_probs

                available_actions[make_action_id] = True
                state[~state] = available_actions.clone()

                # 当遍历到check_budget_list中时，计算预测子图
                if (budget + 1) in check_budget_list:
                    check_idx = check_budget_list.index(budget + 1)
                    subgraph = relabel_graph(graph, state)
                    subgraph_pred = model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)
                    # 解释子图得出的预测类别和真实标签一致；acc_count_list[check_id]累加
                    acc_count_list[check_idx] += (graph.y == subgraph_pred.argmax(dim=1)).sum().item()

                # 达到给定的子图节点数目
                if top_n is not None and budget == top_n - 1:
                    precision_top_n_count += torch.sum(state * graph.ground_truth_mask[0]) / top_n
                    recall_top_n_count += torch.sum(state * graph.ground_truth_mask[0]) / sum(graph.ground_truth_mask[0])

    # 最后位置解释子图就是原图，全部预测正确
    acc_count_list[-1] = num_graph
    # 归一化处理，使acc_count_list中的值在[0, 1]
    acc_count_list = np.array(acc_count_list) / num_graph

    #
    precision_top_n_count = precision_top_n_count / num_graph
    #
    recall_top_n_count = recall_top_n_count / num_graph

    if top_n is not None:
        print('\nACC-AUC: %.4f, Precision@5: %.4f, Recall@5: %.4f' %
              (acc_count_list.mean(), precision_top_n_count, recall_top_n_count))
    else:
        print('\nACC-AUC: %.4f' % acc_count_list.mean())
    print(acc_count_list)

    return np.array(acc_count_list)


# 遍历所有解释子图大小valid_budget；随机生成对应的误差
def bias_detector(model, graph, valid_budget):
    pred_bias_list = []  # 存储对应于valid_budget的误差

    for budget in range(valid_budget):
        num_repeat = 2  # 重复2次
        i_pred_bias = 0  # 第i次的误差

        for i in range(num_repeat):
            bias_selection = torch.zeros(graph.num_edges, dtype=torch.bool)

            cand_action_batch = graph.batch[graph.edge_index[0]]  # 获取候选动作的批次
            cand_action_probs = torch.rand(cand_action_batch.size()).to(device)  # 随机生成候选动作概率
            _, added_actions = scatter_max(cand_action_probs, cand_action_batch)

            bias_selection[added_actions] = True
            bias_subgraph = relabel_graph(graph, bias_selection)  # 生成对应的解释子图
            bias_subgraph_pred = model(bias_subgraph.x, bias_subgraph.edge_index,
                                       bias_subgraph.edge_attr, bias_subgraph.batch).detach()

            i_pred_bias += bias_subgraph_pred / num_repeat

        pred_bias_list.append(i_pred_bias)

    return pred_bias_list


# 计算现阶段解释子图对应的奖励
def get_reward(full_subgraph_pred, new_subgraph_pred, target_y, predicted_labels, pre_reward, model='mutual_info'):
    reward = 0

    if model in ['mutual_info']:
        reward = torch.sum(full_subgraph_pred * torch.log(new_subgraph_pred + EPS), dim=1)
        reward += 2 * (target_y == new_subgraph_pred.argmax(dim=1)).float() - 1.

    elif model in ['binary']:
        reward = (target_y == predicted_labels).float()
        reward = 2. * reward - 1.

    elif model in ['cross_entropy']:
        reward = torch.log(new_subgraph_pred + EPS)[:, target_y]

    reward += 0.9 * pre_reward

    return reward


def train_policy(explainer, model, train_loader, test_loader, optimizer,
                 top_k_ration=0.1, debias_flag=False, top_n=None, batch_size=32, path=None):
    num_episodes = 3  # 训练的总轮次
    train_steps = len(train_loader) * num_episodes
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_steps, eta_min=1e-6)
    # 保存最佳模型的路径
    best_explainer_path = path

    # 在训练开始前测试模型在测试集上的表现
    test_policy_all_with_gnd(explainer, model, test_loader, top_n)

    previous_params = [param.clone() for param in explainer.parameters()]
    ep = 0

    best = 0

    # 存储训练前后基线奖励列表
    previous_baseline_list = []
    current_baseline_list = []

    while ep < num_episodes:
        explainer.train()
        model.eval()

        loss = 0
        avg_reward = []  # 初始化平均奖励列表

        # 遍历训练数据集
        for graph in tqdm(iter(train_loader), total=len(train_loader)):
            graph = graph.to(device)

            # graph = spectral_cluster(graph, model, n_clusters=2).to(device)
            graph = community_cluster(graph, model).to(device)

            valid_budgets = []
            if top_k_ration < 1:
                valid_budget = max(int(top_k_ration * graph.num_edges / batch_size), 1)
                # for data in graph.to_data_list():
                #     n = data.edge_index.size(1)
                #     top_k = int(n // 2 * top_k_ration)
                #     valid_budget = min(top_k, n // 2)
                #     valid_budgets.append(valid_budget)
                # valid_budget = max(int(torch.min(torch.tensor(valid_budgets))), 1)
            else:
                valid_budget = top_k_ration

            batch_loss = 0  # 初始化批次损失

            # 获取完整图的预测结果，并进行 softmax 处理
            full_subgraph_pred = functional.softmax(model(graph.x, graph.edge_index,
                                                          graph.edge_attr, graph.batch), dim=-1).detach()

            current_state = torch.zeros(graph.num_edges, dtype=torch.bool)

            if debias_flag:
                pred_bias_list = bias_detector(model, graph, valid_budget)

            pre_reward = torch.zeros(graph.y.size()).to(device)
            num_beam = 1

            reward_batch = []

            # pre_prob_1 = torch.empty(0, 0)
            # pre_prob_2 = torch.empty(0, 0)

            for budget in range(valid_budget):
                candidate_actions = current_state[~current_state].clone()
                new_state = current_state.clone()

                beam_reward_list = []
                beam_action_list = []

                for beam in range(num_beam):
                    beam_candidate_action = current_state[~current_state].clone()
                    beam_new_state = current_state.clone()

                    if beam == 0:
                        _, added_action_probs, added_actions, unique_batch = explainer.forward_pro(graph, current_state,
                                                                                                   train_flag=False)
                        # pre_prob_1 = added_action_probs
                    else:
                        _, added_action_probs, added_actions, unique_batch = explainer.forward_pro(graph, current_state,
                                                                                                   train_flag=True)
                        # pre_prob_2 = added_action_probs

                    reward_batch = unique_batch

                    beam_candidate_action[added_actions] = True
                    beam_new_state[~current_state] = beam_candidate_action

                    new_subgraph = relabel_graph(graph, beam_new_state)
                    new_subgraph_pred = model(new_subgraph.x, new_subgraph.edge_index,
                                              new_subgraph.edge_attr, new_subgraph.batch)

                    new_subgraph_node = torch.unique(new_subgraph.edge_index)

                    predicted_labels = new_subgraph_pred.argmax(dim=1)

                    if torch.any(added_actions == -1):
                        lack_indices = torch.nonzero(added_actions == -1, as_tuple=False).squeeze()

                        mask = added_actions != -1
                        valid_actions = added_actions[mask].long()
                        selected_edges = graph.edge_index[:, valid_actions]
                        start_node_labels = graph.node_labels[selected_edges[0]]
                        end_node_labels = graph.node_labels[selected_edges[1]]
                        selected_labels = (start_node_labels + end_node_labels) / 2

                        selected_labels = insert_multiple_positions(selected_labels, lack_indices.tolist(), -1)

                        # if predicted_labels.size(0) < graph.y.size(0):
                        #     predicted_labels = insert_multiple_positions(predicted_labels, lack_indices.tolist(), -1)

                    else:
                        selected_edges = graph.edge_index[:, added_actions]
                        start_node_labels = graph.node_labels[selected_edges[0]]
                        end_node_labels = graph.node_labels[selected_edges[1]]
                        selected_labels = (start_node_labels + end_node_labels) / 2

                    comparison = torch.where(selected_labels == -1,
                                             torch.tensor(-1, device=device),  # 如果 selected_labels 是 -1，则结果为 0
                                             torch.where(predicted_labels == selected_labels,
                                                         torch.tensor(1, device=device),  # 如果相同，结果为 1
                                                         torch.tensor(-1, device=device)  # 如果不同，结果为 -1
                                                         )).to(device)

                    if debias_flag:
                        new_subgraph_pred = functional.softmax(new_subgraph_pred - pred_bias_list[budget]).detach()
                    else:
                        new_subgraph_pred = functional.softmax(new_subgraph_pred, dim=-1).detach()

                    reward = get_reward(full_subgraph_pred, new_subgraph_pred, graph.y, predicted_labels,
                                        pre_reward=pre_reward, model='binary')
                    reward = reward + k * comparison
                    reward = reward[unique_batch]

                    if len(previous_baseline_list) - 1 < budget:
                        baseline_reward = 0
                    else:
                        baseline_reward = previous_baseline_list[budget]

                    if len(current_baseline_list) - 1 < budget:
                        current_baseline_list.append([torch.mean(reward)])
                    else:
                        current_baseline_list[budget].append(torch.mean(reward))

                    reward -= baseline_reward

                    added_action_probs = functional.relu(added_action_probs)

                    batch_loss += torch.mean(- torch.log(added_action_probs + EPS) * reward)
                    # batch_loss += torch.mean(torch.pow(- torch.log(added_action_probs + EPS) - reward, 2))
                    # batch_loss += torch.mean(- torch.log(added_action_probs + EPS) * reward +
                    #                          torch.sum(full_subgraph_pred - new_subgraph_pred) ** 2)
                    avg_reward += reward.tolist()

                    beam_reward_list.append(reward)
                    beam_action_list.append(added_actions)

                beam_reward_list = torch.stack(beam_reward_list).T
                beam_action_list = torch.stack(beam_action_list).T

                # beam_reward_list = torch.stack(beam_reward_list).mT
                # beam_action_list = torch.stack(beam_action_list).mT

                batch_loss = batch_loss / num_beam

                max_reward, max_reward_idx = torch.max(beam_reward_list, dim=1)
                max_actions = beam_action_list[range(beam_action_list.size()[0]), max_reward_idx]

                candidate_actions[max_actions] = True
                new_state[~current_state] = candidate_actions

                current_state = new_state.clone()
                pre_reward[reward_batch] = max_reward

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            # tmp = scheduler.get_last_lr()

            # for param in explainer.parameters():
            #     if param.grad is not None:
            #         print(param.grad)

            loss += batch_loss

        loss = loss / (len(train_loader) * valid_budget)
        avg_reward = torch.mean(torch.FloatTensor(avg_reward))
        last_step_reward = avg_reward

        ep += 1
        print('Episode: %d, loss: %.4f, average rewards: %.4f, lr: %f' % (ep, loss.detach(), avg_reward.detach(), scheduler.get_last_lr()[0]))

        if parameters_changed(explainer, previous_params):
            print(f'Parameters changed after episode {ep}')
        else:
            print(f'Parameters did not change after episode {ep}')

        previous_params = [param.clone() for param in explainer.parameters()]

        explainer.train()

        account_list = test_policy_all_with_gnd(explainer, model, test_loader, top_n)
        average = sum(account_list) / len(account_list)
        if account_list[0] > best:
            best = account_list[0]
            torch.save({
                'epoch': ep,
                'explainer_state_dict': explainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc_count_list': account_list,
            }, best_explainer_path)

        if ep == num_episodes:

            # 在训练完成后，加载并输出最佳模型的信息
            checkpoint = torch.load(best_explainer_path)
            print(f'Best Epoch: {checkpoint["epoch"]}, Best Account List: {checkpoint["acc_count_list"]}')

        previous_baseline_list = [torch.mean(torch.stack(baseline)) for baseline in current_baseline_list]
        current_baseline_list = []

    return explainer
















