'''
读取仿真avi输出
构造模型训练数据
'''
import os
import time
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import itertools
import seaborn as sns
import geopandas as gpd
import torch
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# 读取AVI检测器检测数据xml(没用了)
def load_avi_data(file_path):
    data = []
    context = ET.iterparse(file_path, events=("start", "end"))
    for event, elem in context:
        if event == "end" and elem.tag == "instantOut":
            if elem.get('state') == 'enter':
                entry = {
                    'id': elem.get('id'),
                    'time': float(elem.get('time')),
                    'vehID': elem.get('vehID')
                }
                data.append(entry)
            elem.clear()
    avidata = pd.DataFrame(data)
    avidata['id'] = avidata['id'].astype(int)
    avidata['time'] = avidata['time'].astype(float)
    return avidata

def load_network_data():
    nodes = gpd.read_file('data/taicangNet/nodes.shp', encoding='utf-8')
    links = gpd.read_file('data/taicangNet/links.shp', encoding='utf-8')
    links = links[~links['geometry'].isna()]
    links['length'] = links['geometry'].length

    nodes.crs = 'epsg:3857'
    links.crs = 'epsg:3857'

    taicang = gpd.read_file('data/taicangNet/tcborder.shp', encoding='utf-8')

    # 将点边表转换为networkx图
    def gdf2graph(nodes, links):
        links['key'] = 0
        nodes['x'] = nodes['geometry'].x
        nodes['y'] = nodes['geometry'].y
        nodes.set_index('id',inplace=True)
        links.set_index(['u','v','key'],inplace=True)
        return ox.graph_from_gdfs(nodes,links)
    # 生成路网图以检查连通性
    G = gdf2graph(nodes.copy(), links.copy())
    # 最大连通分量
    largest_component = max(list(nx.weakly_connected_components(G)), key=len)
    nodes = nodes[nodes['id'].isin(largest_component)]
    links = links[(links['u'].isin(largest_component)) & (links['v'].isin(largest_component))]
    return nodes, links, taicang
    
# 读取E1_info.xml，获取AVI检测器与link的映射关系
def load_avi_link_map(E1_info_path):
    tree = ET.parse(E1_info_path)
    root = tree.getroot()
    data = []
    for E1 in root.findall('instantInductionLoop'):
        entry = {
            'id': E1.get('id'),
            'link': int(float(E1.get('link'))),
        }
        data.append(entry)
    return pd.DataFrame(data)['link']




# 最大流量生成树算法(MFST)
def MFST(Gd, AVINUM):
    parent = {node: node for node in Gd.nodes}

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1, root2 = find(node1), find(node2)
        if root1 != root2:
            parent[root2] = root1
            return True
        return False

    T = nx.Graph()
    T.add_nodes_from(range(AVINUM))
    L_star = []
    sorted_edges = sorted(Gd.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

    for u, v, attr in sorted_edges:
        if T.degree(u) < 2 and T.degree(v) < 2 and union(u, v):
            T.add_edge(u, v, weight=attr['weight'])
            L_star.append((u, v))
    return L_star, T


# 匹配流量\匹配行程时间\断面流量计算
def get_match_flow_time(avidata, aviseq, AVINUM, just_match=False):
    weights = np.zeros([AVINUM, AVINUM])
    times = np.zeros([AVINUM, AVINUM])
    if just_match:
        # 如果只需要匹配流量，不需要计算行程时间
        for vehID, group in avidata.groupby('vehID'):
            avi_seq = list(group['link_reset'])
            combinations = list(itertools.combinations(avi_seq, 2))
            for comb in combinations:
                weights[comb[0], comb[1]] += 1
        match_flows = weights[np.ix_(aviseq, aviseq)]
        return match_flows, None, None
    else:
        # veh = avidata_group['vehID'].to_numpy()
        # link = avidata_group['link_reset'].to_numpy()
        # ts   = avidata_group['time'].to_numpy()
        # 如果需要匹配流量和行程时间
        for vehID, group in avidata.groupby('vehID'):
            avi_seq = list(group['link_reset'])
            combinations = list(itertools.combinations(avi_seq, 2))
            for comb in combinations:
                weights[comb[0], comb[1]] += 1
                times[comb[0], comb[1]] += group.loc[group['link_reset'] == comb[1], 'time'].values[0] - group.loc[group['link_reset'] == comb[0], 'time'].values[0]
        mask = weights != 0
        times[mask] = times[mask] / weights[mask]
        match_flows = weights[np.ix_(aviseq, aviseq)]
        match_times = times[np.ix_(aviseq, aviseq)]
        id_counts = avidata['id'].value_counts()
        sorted_counts = [id_counts.get(id_val, 0) for id_val in aviseq]
        section_flow = np.tile(np.array(sorted_counts).reshape(AVINUM, 1), (1, AVINUM))
        return match_flows, match_times, section_flow

def get_match_flow_time_fast(avidata, aviseq, AVINUM, dedup_first_occurrence=True):
    """
    avidata: DataFrame，至少包含 ['vehID', 'link_reset', 'time', 'id']
    aviseq: 需要抽取的节点索引列表/ndarray（应在 [0, AVINUM-1] 范围内）
    AVINUM: 节点总数（决定输出矩阵大小）
    dedup_first_occurrence: 是否在每辆车内部按 link 只保留“首次出现”的时间（强烈建议开）
    """
    # 预分配
    weights = np.zeros((AVINUM, AVINUM), dtype=np.int32)
    times   = np.zeros((AVINUM, AVINUM), dtype=np.float64)

    # 1) 按 vehID, time 排序，拉成 numpy，避免 groupby 循环开销
    df = avidata.sort_values(['vehID', 'time'], kind='stable')
    veh  = df['vehID'].to_numpy()
    link = df['link_reset'].to_numpy()
    t    = df['time'].to_numpy().astype(np.float64, copy=False)

    # 2) 找各车辆分段边界
    uniq_veh, first_idx = np.unique(veh, return_index=True)
    bounds = np.append(first_idx, len(veh))

    for i in range(len(uniq_veh)):
        s, e = bounds[i], bounds[i+1]
        L = e - s
        if L < 2:
            continue

        seq = link[s:e]
        tt  = t[s:e]

        if dedup_first_occurrence:
            # 保留同一 link 的首次出现（最早时间）——降低配对数量，避免一车内重复环
            # np.unique 返回按出现顺序的“首次索引”
            uniq_links, first_occ_idx = np.unique(seq, return_index=True)
            # 需要按出现顺序排序 first_occ_idx（unique 不保证有序）
            order = np.sort(first_occ_idx)
            seq = seq[order]
            tt  = tt[order]
            L   = len(seq)
            if L < 2:
                continue

        # 3) 向量化生成 i<j 的索引对
        I, J = np.triu_indices(L, k=1)
        src = seq[I]                    # 左上角索引
        dst = seq[J]                    # 右下角索引
        dt  = tt[J] - tt[I]             # 时间差

        # 4) 原地累加（允许重复索引）
        np.add.at(weights, (src, dst), 1)
        np.add.at(times,   (src, dst), dt)

    # 5) 求平均时间（仅对出现过的位置）
    mask = weights > 0
    times[mask] = times[mask] / weights[mask]

    # 6) 抽取所需子矩阵
    aviseq = np.asarray(aviseq, dtype=int)
    match_flows = weights[np.ix_(aviseq, aviseq)]
    match_times = times[np.ix_(aviseq, aviseq)]

    # 7) section_flow（与你原逻辑等价，但更直接）
    id_counts = df['id'].value_counts()
    sorted_counts = np.array([id_counts.get(int(id_val), 0) for id_val in aviseq], dtype=np.float32)
    section_flow = np.repeat(sorted_counts[:, None], AVINUM, axis=1)

    return match_flows, match_times, section_flow

def read_avi_csv(csv_path, avi_link_map, yureqi=0, AVINUM=200):
    # 加载AVI数据
    # avidata = load_avi_data('net_data/output_E1.xml')
    avidata = pd.read_csv(csv_path)
    avidata['id'] = avidata['id'].astype(int)
    avidata['time'] = avidata['time'].astype(float)
    avidata['vehID'] = avidata['vehID'].astype(str)
    avidata['vtype'] = avidata['vehID'].str[0]

    avidata['link'] = avidata['id'].map(avi_link_map)
    avidata = avidata[avidata['link'].isin(avidata['link'].value_counts().head(AVINUM).index)]
    avidata = avidata[avidata['time'] > yureqi]  # 过滤掉前30分钟预热期数据
    avidata['k'] = ((avidata['time']-yureqi) // 3600).astype(int)
    avidata['link_reset'] = pd.factorize(avidata['link'])[0]
    print(f"有数据的AVI检测器所在路段数量: {len(avidata['link'].unique())}")
    # assert AVINUM == len(avidata['link'].unique()) # 统计AVI检测器数量
    return avidata

def plot_data_distribution(avidata):
    # 绘制数据分布图
    plt.figure(figsize=(12, 6))
    count_data = avidata['link_reset'].value_counts().sort_index()
    plt.plot(count_data.index, count_data.sort_values(ascending=False).values)
    plt.title('AVI Data Distribution by Link Reset')
    plt.xlabel('Link Reset ID')
    plt.ylabel('Count')
    # 只显示整百的标签
    max_label = avidata['link_reset'].max()
    ticks = [i for i in range(0, max_label + 1) if i % 100 == 0]
    plt.xticks(ticks=ticks)
    plt.tight_layout()
    plt.savefig('avi_data_distribution.png')
    # plt.show()
    
def plot_time_distribution(avidata):
    # 绘制时间分布图
    plt.figure(figsize=(12, 6))
    sns.histplot(avidata['time'], bins=100, kde=True)
    plt.title('AVI Data Time Distribution')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('avi_time_distribution.png')
    # plt.show()

def plot_link_distribution(avidata):
    # 绘制link分布图
    plt.figure(figsize=(12, 6))
    count_data = avidata['link'].value_counts().sort_index()
    plt.plot(count_data.index, count_data.values)
    plt.title('AVI Data Distribution by Link')
    plt.xlabel('Link ID')
    plt.ylabel('Count')
    # 只显示整百的标签
    max_label = avidata['link'].max()
    ticks = [i for i in range(0, max_label + 1) if i % 100 == 0]
    plt.xticks(ticks=ticks)
    plt.tight_layout()
    plt.savefig('avi_link_distribution.png')
    # plt.show()

def get_aviseq(avidata, AVINUM):
    # avidata = avidata[avidata['link'].isin(avidata['link'].value_counts().head(AVINUM).index)] # 只保留前200个link的AVI数据
    print("正在计算AVI序列..")
    match_flows, match_times, section_flow = get_match_flow_time(avidata, list(range(AVINUM)), AVINUM, just_match=True)

    # 构建初始无权重图用于MFST
    G_init = nx.Graph()
    G_init.add_nodes_from(range(AVINUM))
    for i in range(AVINUM):
        for j in range(i + 1, AVINUM):
            G_init.add_edge(i, j, weight=match_flows[i, j])

    # 应用MFST算法重排AVI序列
    L_star, T_star = MFST(G_init, AVINUM)
    aviseq = list(nx.dfs_preorder_nodes(T_star))

    # match_flows = match_flows[np.ix_(aviseq, aviseq)]
    # match_times = match_times[np.ix_(aviseq, aviseq)]
    # section_flow = section_flow[np.ix_(aviseq, aviseq)]
    # avi_tensor = np.stack((match_flows, match_times, section_flow), axis=-1)
    # print(f"AVI tensor数据形状: {avi_tensor.shape}, aviseq长度: {len(aviseq)}")
    return aviseq

def generate_avi_net(match_time_flow, aviseq):
    '''
    暂时不写这个函数，看看图神经网络要求的输入是什么样子
    '''
    match_flows, match_times, section_flow = match_time_flow
    # 构建有向图并添加权重
    G = nx.DiGraph()
    for i in aviseq:
        G.add_node(i, section_flow=section_flow[i, i])
    for i in aviseq:
        for j in aviseq:
            if i != j:
                G.add_edge(
                    i, j,
                    flow_weight=match_flows[i, j],
                    time_weight=match_times[i, j]
                )
    return G

def generate_avi_tensor(avidata, aviseq, AVINUM, windows_size=2):
    # 重新计算分组后的match_flows, match_times, section_flow
    print("正在生成AVI tensor数据...")
    t1 = time.time()
    grouped_data = []
    unique_k_values = avidata['k'].unique()
    for start_k in range(unique_k_values.min(), unique_k_values.max() - windows_size + 1):
        selected_rows = avidata[(avidata['k'] >= start_k) & (avidata['k'] < start_k + windows_size)]
        grouped_data.append(selected_rows)

    # AVI观测量加载
    tensors_perslot = []
    for avidata2hour in grouped_data:
        print(f"正在处理时间段: {avidata2hour['k'].min()} 到 {avidata2hour['k'].max()}")
        match_flows, match_times, section_flow = get_match_flow_time_fast(avidata2hour, aviseq, AVINUM)
        tensor_perslot = np.stack((match_flows, match_times, section_flow), axis=0).astype(np.float32)
        tensors_perslot.append(tensor_perslot)
    tensors_perslot = np.stack(tensors_perslot, axis=0)  # (num_slots, 3, AVINUM, AVINUM)
    t2 = time.time()
    print(f"AVI tensor数据形状: {tensors_perslot.shape}, 耗时: {t2 - t1:.2f}秒")
    return tensors_perslot

def generate_samples(tensors_perslot, tps=8):
    # 每8小时一个样本
    samples = []
    for start in range(tensors_perslot.shape[0] - tps + 1):
        sample = np.stack(tensors_perslot[start:start + tps], axis=0)
        samples.append(sample)
    samples = np.stack(samples, axis = 0)
    # np.save('samples.npy', samples.astype(np.float32))
    return samples
    
def generate_labels(flow_label_paths, tps=8):
    # 标签数据构建
    labels = []
    flow_data_list = []
    path_set = []
    for flow_label_path in flow_label_paths:
        flow_data = []
        for event, elem in ET.iterparse(flow_label_path, events=("start",)):
            if elem.tag == "flow":
                rs = elem.get("rs")
                flow_id = elem.get("id")
                flow_number = elem.get("number")
                vtype = elem.get("type")
                flow_via = elem.get('via')
                flow_begin = elem.get("begin")
                if flow_id and flow_number:
                    flow_data.append({
                        'fid': flow_id,
                        'number': int(flow_number),
                        'vtype': str(vtype),
                        'via': str(flow_via),
                        'beginhour': int(flow_begin) / 3600,
                        'rs': rs
                    })
                elem.clear()
        flow_data_df = pd.DataFrame(flow_data)
        path_set.append(flow_data_df['rs'].unique().tolist())
        flow_data_list.append(flow_data_df)


    labels = []
    for start in range(24-tps):
        label = []
        for i, flow_data in enumerate(flow_data_list):
            fs = flow_data[(flow_data['beginhour'] >= start) & (flow_data['beginhour'] < start + tps)]
            rs_list = path_set[i]
            # fs = flow_data['rs'].value_counts().sort_index()
            counts = fs.groupby('rs')['number'].sum()
            label.append([counts.get(rs, 0) for rs in rs_list])
            # label.append(fs.values)
        label = np.stack(label, axis=0)
        # print(label)
        labels.append(label)
    labels = np.stack(labels, axis=0)  # (num_slots, num_rs)
    labels = labels.astype(np.float32)
    print(f"生成标签数据，形状: {labels.shape}") 
    return labels
        

if __name__ == "__main__":
    avi_link_map = load_avi_link_map('data/sumonet/E1_info.xml')
    AVINUM = avi_link_map.nunique()
    print(f"AVINUM = {AVINUM}")
    
    AVINUM = 256  # 设置AVI检测器数量
    TPS = 4
    samples_all = []
    labels_all = []
    days = 7 * 30
    start = 0
    From_mid = False  # 是否从中间开始
    
    aviseq = []
    if From_mid:
        start = 70
        # 从第start天开始处理数据
        samples_all.append(np.load('dataset/samples_20250820.npz')['samples'])
        labels_all.append(np.load('dataset/labels_20250820.npz')['labels'])
        print(f"从第{start+1}天开始处理数据，样本形状: {samples_all[0].shape}, 标签形状: {labels_all[0].shape}")
    
    # try:
    for day in range(start, days):  # 1到28天
        print(f"==================正在处理第{day+1}天的数据===================")
        avicsv_path = f'data/sumonet/sumo_avi_20250820/sumo_avi_data_{day}.csv'
        flow_label_paths = [f'data/sumonet/heavy_flow/heavy_truck_{day}.rou.xml', f'data/sumonet/light_flow/light_truck_{day}.rou.xml']
    
        avidata = read_avi_csv(avicsv_path, avi_link_map, yureqi=0,AVINUM=AVINUM)
        print(f"AVI数据行数: {len(avidata)}, AVINUM: {AVINUM}")
    
        
        # 生成AVI tensor数据
        if day == start :   
            # plot_data_distribution(avidata)
            # plot_time_distribution(avidata)
            # plot_link_distribution(avidata) 
            aviseq = get_aviseq(avidata, AVINUM)
            print(f"AVI序列长度: {len(aviseq)}")
        samples = generate_avi_tensor(avidata, aviseq, AVINUM)
        samples_all.append(samples)
        
        # print("AVI tensor数据生成完成")
        labels = generate_labels(flow_label_paths, tps=TPS)
        labels_all.append(labels)
    
    # 合并所有天的samples并转换为torch.tensor
    samples_all = np.stack(samples_all, axis=0)  # (num_days, numslots, 3, AVINUM, AVINUM)
    samples_all = torch.tensor(samples_all, dtype=torch.float32)
    labels_all = np.concatenate(labels_all, axis=0)
    labels_all = torch.tensor(labels_all, dtype=torch.float32)
    print(f"所有天的样本数据形状: {samples_all.shape}")   # (num_days, 23, 3, AVINUM, AVINUM)
    print(f"所有天的标签数据形状: {labels_all.shape}")
    
    save_path='.\dataset'
    os.makedirs(save_path, exist_ok=True)
        
    # np.savez_compressed(f"{save_path}/samples_{time.strftime('%Y%m%d')}.npz", samples=samples_all.numpy().astype(np.float32))
    # np.savez_compressed(f"{save_path}/labels_{time.strftime('%Y%m%d')}.npz", labels=labels_all.numpy().astype(np.float32))
    np.save(f"{save_path}/samples_{time.strftime('%Y%m%d')}.npy", samples_all.numpy().astype(np.float16))
    np.save(f"{save_path}/labels_{time.strftime('%Y%m%d')}.npy", labels_all.numpy().astype(np.float16))
    print(f"样本数据已保存到 {save_path}/samples_{time.strftime('%Y%m%d')}.npz")
        
    # except Exception as e:
    #     print(f"处理数据时发生错误: {e}")
    #     samples_all = np.concatenate(samples_all, axis=0)
    #     samples_all = torch.tensor(samples_all, dtype=torch.float32)
    #     labels_all = np.concatenate(labels_all, axis=0)
    #     labels_all = torch.tensor(labels_all, dtype=torch.float32)
    #     print(f"所有天的样本数据形状: {samples_all.shape}")   # (num_samples, tps, 3, AVINUM, AVINUM)
    #     print(f"所有天的标签数据形状: {labels_all.shape}, uniques: {len(uniques)}")
    #     save_path='.\dataset'
    #     os.makedirs(save_path, exist_ok=True)
    #     np.savez_compressed(f"{save_path}/samples_{time.strftime('%Y%m%d')}_error.npz", samples=samples_all.numpy().astype(np.float32))
    #     np.savez_compressed(f"{save_path}/labels_{time.strftime('%Y%m%d')}_error.npz", labels=labels_all.numpy().astype(np.float32))
    #     print(f"错误数据已保存到 {save_path}/samples_{time.strftime('%Y%m%d')}_error.npz 和 {save_path}/labels_{time.strftime('%Y%m%d')}_error.npz")
    