import torch

from torch_geometric.loader.dataloader import DataLoader

from Exist_Model.data_loader_pool.muatg_dataloader import Mutagen
from Exist_Model.data_loader_pool.nci1_dataloader import Nci1
from Exist_Model.data_loader_pool.bzr_dataloader import Bzr

from Exist_Model.gnn_model_pool.mutag_gine import MutagNet
from Exist_Model.gnn_model_pool.nci1_gcn import Nci1Net
from Exist_Model.gnn_model_pool.bzr_gcn import BzrNet

from utils import set_seed, evaluate_metrics
from utils.reorganizer import filter_correct_data_batch
from Ours import ExplainerBatchGraphTransform
from train_test_pool import train_policy, test_policy_all_with_gnd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    set_seed(19930819)

    dataset_name = 'nci1'
    _hidden_size = 64
    _num_labels = 2
    debias_flag = False
    top_n = None
    batch_size = 32
    lr = 1e-3

    path = 'params/%s_net.pt' % dataset_name
    train_dataset = Nci1('Data/Nci1', mode='training')
    test_dataset = Nci1('Data/Nci1', mode='testing')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    model = torch.load(path).to(device)
    model.eval()

    model_explain = torch.load(path).to(device)
    model_explain.eval()

    # 将model中预测正确的数据筛选出来组建新的dataset和loader
    train_dataset, train_loader = filter_correct_data_batch(model, train_dataset, train_loader,
                                                            'training', batch_size=batch_size)
    test_dataset, test_loader = filter_correct_data_batch(model, test_dataset, test_loader,
                                                          'testing', batch_size=batch_size)

    explainer = ExplainerBatchGraphTransform(_model=model_explain, _num_labels=_num_labels,
                                             _hidden_size=_hidden_size, _use_edge_attr=False).to(device)
    # model_dict = torch.load('explainer_params/nci1/nci1_merge0.933_k0.5_com.pth')
    # explainer.load_state_dict(model_dict["explainer_state_dict"])

    pro_flag = False
    optimizer = explainer.get_optimizer(lr=lr, scope='all')
    # optimizer.load_state_dict(model_dict["optimizer_state_dict"])

    top_k_ration = 0.1
    train_policy(explainer, model, train_loader, test_loader, optimizer, top_k_ration,
                 debias_flag=debias_flag, top_n=top_n, batch_size=batch_size,
                 path='explainer_params/nci1/nci1_merge_k0.5_com_4.pth')

    # acc_storage_path = 'NEW_Result/Cluster/nci1_acc_KM.txt'
    # fid_storage_path = 'NEW_Result/Cluster/nci1_fid_KM.txt'
    # infid_storage_path = 'NEW_Result/Cluster/nci1_infid_KM.txt'
    #
    # explainer.eval()
    # with torch.no_grad():
    #     acc_count_list, fid_count_list, infid_count_list = evaluate_metrics(explainer, model, test_loader)
    #
    #     with open(acc_storage_path, 'w') as f:
    #         for count in acc_count_list:
    #             f.write(f"{count}\n")
    #
    #     with open(fid_storage_path, 'w') as f:
    #         for count in fid_count_list:
    #             f.write(f"{count}\n")
    #
    #     with open(infid_storage_path, 'w') as f:
    #         for count in infid_count_list:
    #             f.write(f"{count}\n")



