import torch
from torch_geometric.loader.dataloader import DataLoader
import rdkit

from rdkit import Chem
# from rdkit.Chem import Draw
# from rdkit.Chem.Draw import rdMolDraw2D

from Exist_Model.data_loader_pool.muatg_dataloader import Mutagen
from Exist_Model.gnn_model_pool.mutag_gine import MutagNet
from Ours import ExplainerBatchGraphTransform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

elements = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]
valence_map = {0: "-", 1: "=", 2: "#"}

data = Mutagen('Data/MUTAG', mode='testing')
loader = DataLoader(data, batch_size=1, shuffle=False)

graph_data = []

for graph in iter(loader):
    if graph.y == 1:
        graph_data.append(graph)
print_data = graph_data[30]

edge_index = print_data.edge_index  # (2, num_edges) 的张量，表示边的起始和结束节点
node_features = print_data.x        # (num_nodes, num_features) 的张量，表示节点特征
node_labels = []
edge_attr = print_data.edge_attr
graph_labels = print_data.y          # (num_nodes,) 的张量，表示节点标签（可选）

model = torch.load('params/mutag_net.pt').to(device)
model.eval()

checkpoint = torch.load('explainer_params/mutag/mutag_merge_k1_KM0.958.pth', map_location=torch.device('cpu'))
explainer = ExplainerBatchGraphTransform(_model=model, _num_labels=2,
                                         _hidden_size=32, _use_edge_attr=False).to(device)
model_params = checkpoint['explainer_state_dict']
explainer.load_state_dict(model_params)
explainer.eval()

list = []

with torch.no_grad():
    print_data = print_data.to(device)
    state = torch.zeros(print_data.num_edges, dtype=torch.bool)
    for budget in range(15):
        available_actions = state[~state].clone()
        cand_action_probs, make_action_probs, make_action_id, _ = explainer.forward_pro(graph=print_data, state=state,
                                                                                        train_flag=False)
        available_actions[make_action_id] = True
        state[~state] = available_actions.clone()
        list.append(make_action_id)

list = [tensor.item() for tensor in list]
columns = [edge_index[:, idx] for idx in list]

print(columns)
print(node_features)

h_index = elements.index("H")
filtered_columns = [
    edge for edge in columns
    if h_index not in (torch.argmax(node_features[edge[0]]).item(), torch.argmax(node_features[edge[1]]).item())
]
print("Filtered edges and corresponding node elements:")
for edge in filtered_columns:
    node1_idx, node2_idx = edge.tolist()
    node1_element = elements[torch.argmax(node_features[node1_idx]).item()]
    node2_element = elements[torch.argmax(node_features[node2_idx]).item()]
    print(f"Edge: {edge.tolist()} -> Nodes: {node1_element} - {node2_element}")


# 1. 构建 RDKit 分子
mol = Chem.RWMol()  # 使用可修改的分子对象

# 2. 添加节点（原子）到分子中
for feature in node_features:
    # 选择非零的特征（即哪种元素）
    idx = feature.argmax().item()  # 获取最大值的索引
    atom = Chem.Atom(elements[idx])  # 创建原子对象
    mol.AddAtom(atom)  # 添加到分子中

# 3. 添加边（键）到分子中
for i in range(edge_index.shape[1]):
    atom1 = edge_index[0, i].item()  # 起始原子索引
    atom2 = edge_index[1, i].item()  # 目标原子索引
    valence = edge_attr[i].argmax().item()  # 获取最大值的索引，表示边的 valence

    # 根据 valence 设置键类型
    if valence == 0:
        bond_type = Chem.BondType.SINGLE  # 单键
    elif valence == 1:
        bond_type = Chem.BondType.DOUBLE  # 双键
    elif valence == 2:
        bond_type = Chem.BondType.TRIPLE  # 三键
    else:
        bond_type = Chem.BondType.SINGLE  # 默认使用单键，作为备用

    # 检查该边是否已经存在
    bond_exists = mol.GetBondBetweenAtoms(atom1, atom2) is not None
    if not bond_exists:
        # 添加键到分子中
        mol.AddBond(atom1, atom2, bond_type)

# 4. 转换为 SMILES 字符串
smiles = Chem.MolToSmiles(mol)

print("SMILES:", smiles)

#
# # 将此SMILES字符串替换为您的分子SMILES字符串
# smiles_string = smiles
#
# # 从SMILES创建分子对象
# molecule = Chem.MolFromSmiles(smiles_string)
#
# # 为分子生成2D坐标
# Chem.rdDepictor.Compute2DCoords(molecule)
#
# # 创建一个用于绘制分子的对象
# drawer = Draw.MolDraw2DSVG(400, 400)
#
# # 指定要高亮显示的原子的索引
# highlight_atoms = []  # 根据需要替换为正确的原子索引
#
# # 绘制分子结构并高亮显示指定的原子
# drawer.DrawMolecule(
#     molecule,
#     highlightAtoms=highlight_atoms,
#     highlightAtomColors={atom_idx: (0.5, 0.5, 1) for atom_idx in highlight_atoms}
# )
# drawer.FinishDrawing()
#
# # 获取SVG图像文本
# svg = drawer.GetDrawingText()
#
# # 将SVG图像保存到文件
# with open('DDI2.svg', 'w') as svg_file:
#     svg_file.write(svg)
