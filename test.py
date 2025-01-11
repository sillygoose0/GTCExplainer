# import torch
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#
#
# def insert_multiple_positions(tensor, insert_indices, value):
#     if isinstance(insert_indices, list):
#         # 如果 insert_indices 是列表，处理每个索引
#         for i, index in enumerate(insert_indices):
#             index = int(index)  # 确保 index 是整数
#             tensor = torch.cat((tensor[:index + i], torch.tensor([value], device=tensor.device), tensor[index + i:]))
#     elif isinstance(insert_indices, int):
#         # 如果 insert_indices 是单个整数
#         index = int(insert_indices)  # 确保 index 是整数
#         tensor = torch.cat((tensor[:index], torch.tensor([value], device=tensor.device), tensor[index:]))
#
#     return tensor
#
#
# added_actions = torch.tensor([2, 6, 5, 11, -1, 13])
#
# if torch.any(added_actions == -1):
#     lack_indices = torch.nonzero(added_actions == -1, as_tuple=False).squeeze()

    # mask = added_actions != -1
    # valid_actions = added_actions[mask].long()
    # selected_edges = graph.edge_index[:, valid_actions]
    # start_node_labels = graph.node_labels[selected_edges[0]]
    # end_node_labels = graph.node_labels[selected_edges[1]]
    # selected_labels = (start_node_labels + end_node_labels) / 2
    # selected_labels = torch.tensor([1, 0, 1, 1, 0]).to(device)
    #
    # selected_labels = insert_multiple_positions(selected_labels, lack_indices.tolist(), -1)
# else:
    # selected_edges = graph.edge_index[:, added_actions]
    # start_node_labels = graph.node_labels[selected_edges[0]]
    # end_node_labels = graph.node_labels[selected_edges[1]]
    # selected_labels = (start_node_labels + end_node_labels) / 2

# predicted_labels = torch.tensor([1, 0, 1, 1, 0, 1]).to(device)
# comparison = torch.where(selected_labels == -1,
#                          torch.tensor(0, device=device),  # 如果 selected_labels 是 -1，则结果为 0
#                          torch.where(predicted_labels == selected_labels,
#                                      torch.tensor(1, device=device),  # 如果相同，结果为 1
#                                      torch.tensor(-1, device=device)  # 如果不同，结果为 -1
#                                      )).to(device)
# print(comparison)


import torch

from torch_geometric.loader.dataloader import DataLoader

from Exist_Model.data_loader_pool.muatg_dataloader import Mutagen
from Exist_Model.data_loader_pool.nci1_dataloader import Nci1
from Exist_Model.data_loader_pool.aids_dataloader import Aids
from Exist_Model.data_loader_pool.bzr_dataloader import Bzr

from Exist_Model.gnn_model_pool.mutag_gine import MutagNet
from Exist_Model.gnn_model_pool.nci1_gcn import Nci1Net
from Exist_Model.gnn_model_pool.aids_gcn import AidsNet
from Exist_Model.gnn_model_pool.bzr_gcn import BzrNet

from utils import set_seed
from utils.reorganizer import filter_correct_data_batch
from Ours import ExplainerBatchGraphTransform
from train_test_pool import train_policy, test_policy_all_with_gnd
from utils import evaluate_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 加载 .pth 文件
# checkpoint = torch.load('explainer_params/mutag/mutag_merge0.997', map_location=torch.device('cpu'))
#
# print(checkpoint['acc_count_list'])

set_seed(19930819)

dataset_name = 'nci1'
_hidden_size = 64
_num_labels = 2
debias_flag = False
top_n = None
batch_size = 32

path = 'params/%s_net.pt' % dataset_name
test_dataset = Nci1('Data/Nci1', mode='testing')

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = torch.load(path)
model.eval()

model_explain = torch.load(path)
model_explain.eval()

checkpoint = torch.load('explainer_params/nci1/nci1_merge_k0.5_com_4.pth', map_location=torch.device('cpu'))

explainer = ExplainerBatchGraphTransform(_model=model_explain, _num_labels=_num_labels,
                                         _hidden_size=_hidden_size, _use_edge_attr=False).to(device)

test_dataset, test_loader = filter_correct_data_batch(model, test_dataset, test_loader,
                                                      'testing', batch_size=batch_size)

model_params = checkpoint['explainer_state_dict']
acc = checkpoint['acc_count_list']
with torch.no_grad():
    explainer.load_state_dict(model_params)
print(acc)

acc_storage_path = 'NEW_Result/Nci1/melt/GTC/002/acc_2.txt'
fid_storage_path = 'NEW_Result/Nci1/melt/GTC/002/fid_2.txt'
infid_storage_path = 'NEW_Result/Nci1/melt/GTC/002/infid_2.txt'

explainer.eval()
with torch.no_grad():
    acc_count_list, fid_count_list, infid_count_list = evaluate_metrics(explainer, model, test_loader)

    with open(acc_storage_path, 'w') as f:
        for count in acc_count_list:
            f.write(f"{count}\n")

    with open(fid_storage_path, 'w') as f:
        for count in fid_count_list:
            f.write(f"{count}\n")

    with open(infid_storage_path, 'w') as f:
        for count in infid_count_list:
            f.write(f"{count}\n")







