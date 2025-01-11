import torch

from torch_geometric.loader.dataloader import DataLoader

from Exist_Model.data_loader_pool.muatg_dataloader import Mutagen
from Exist_Model.data_loader_pool.nci1_dataloader import Nci1
from Exist_Model.data_loader_pool.molt_dataloader import Molt

from Exist_Model.gnn_model_pool.mutag_gine import MutagNet
from Exist_Model.gnn_model_pool.nci1_gcn import Nci1Net
from Exist_Model.gnn_model_pool.molt_gcn import MoltNet

from utils import set_seed, evaluate_metrics
from utils.reorganizer import filter_correct_data_batch
from Ours import ExplainerBatchGraphTransform
from train_test_pool import train_policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    set_seed(19930819)

    dataset_name = 'mutag'
    _hidden_size = 32
    _num_labels = 2
    debias_flag = False
    top_n = None
    batch_size = 32

    path = 'params/%s_net.pt' % dataset_name
    train_dataset = Mutagen('Data/MUTAG', mode='training')
    test_dataset = Mutagen('Data/MUTAG', mode='testing')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    pro_flag = False
    optimizer = explainer.get_optimizer()

    top_k_ration = 0.1
    train_policy(explainer, model, train_loader, test_loader, optimizer, top_k_ration,
                 debias_flag=debias_flag, top_n=top_n, batch_size=batch_size,
                 path='explainer_params/mutag/mutag_merge_k1_KM.pth')

    # acc_storage_path = 'NEW_Result/Cluster/mutag_acc_louvain.txt'
    # fid_storage_path = 'NEW_Result/Cluster/mutag_fid_louvain.txt'
    # infid_storage_path = 'NEW_Result/Cluster/mutag_infid_louvain.txt'
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


