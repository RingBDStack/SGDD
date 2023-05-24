
'''
Utilities functions for the framework.
'''
import pandas as pd
import numpy as np
import os
import argparse
import torch
import math
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import warnings
import sklearn.metrics
warnings.filterwarnings('ignore')

import pdb

# torch.set_default_tensor_type('torch.DoubleTensor')
# torch_dtype = torch.float64 #torch.float32

res_dir = 'results'
data_dir = 'data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu' #Can set to CPU here for timing comparison

def parse_args():
    parser = argparse.ArgumentParser()
    #
    parser.add_argument("--verbose", dest='verbose', action='store_const', default=False, const=True, help='Print out verbose info during optimization')
    parser.add_argument("--seed", dest='fix_seed', action='store_const', default=False, const=True, help='Fix seed for reproducibility and fair comparison.')
    parser.add_argument("--n_epochs", dest='n_epochs', type=int, default=300, help='Number of COPT iterations during training.')
    parser.add_argument("--hike", dest='lr_hike', action='store_const', default=False, const=True, help='Use learning rate hiking. This is recommended for most applications.')
    parser.add_argument("--hike_interval", dest='hike_interval', type=int, default=15, help='Number of iterations having *low loss* between lr hikes.')
    parser.add_argument("--fast", dest='fast', action='store_const', default=False, const=True, help='Use fast kernel, in particular multiscale laplacian')
    parser.add_argument("--early_stopping", action='store_true', default=False, help='stop training early if loss repeatedly reaches below some threshold, useful for sketching.')
    
    parser.add_argument("--dataset_type", type=str, default='synthetic', help='dataset type, can be "synthetic" or "real"') 
    parser.add_argument("--dataset_name", type=str, default='PROTEINS', help='dataset name, will be ignored if dataset_type is synthetic')  
    parser.add_argument("--m", default=30, type=int, help='Number of vertices in graph X, e.g. query graph, useful especially when testing')
    parser.add_argument("--n", default=30, type=int, help='Number of vertices in graph Y, e.g. dataset graph, useful especially when testing')  

    parser.add_argument("--sinkhorn_iter", type=int, default=10, help='Number of Sinkhorn scaling iterations during optimization')
    parser.add_argument("--grid_search", dest='grid_search', action='store_const', default=False, const=True, help='grid search for learning SVC classifier')
    parser.add_argument("--compress_fac", default=-1, type=int, help='Factor of compression, e.g. 2 means reduce to half as many vertices')
    parser.add_argument("--got_it", dest='st_it', type=int, default=5, help='Number of Sinkhorn iterations for GOT')
    parser.add_argument("--got_tau", dest='st_tau', type=float, default=1, help='Number of Sinkhorn iterations for GOT')
    parser.add_argument("--got_n_sample", dest='st_n_samples', type=int, default=10, help='Number of samples in stochastic sampling for GOT')
    parser.add_argument("--got_n_epochs", dest='st_epochs', type=int, default=1000, help='Number of Sinkhorn iterations for GOT')
    parser.add_argument("--got_lr", dest='st_lr', type=float, default=.5, help='Number of Sinkhorn iterations for GOT')
    parser.add_argument("--gw_alpha", dest='gw_alpha', type=float, default=.8, help='alpha parameter for GW')
    parser.add_argument("--gw_metric", dest='gw_metric', type=str, default='sqeuclidean', help='features metric for GW')    
    
    opt = parser.parse_args()    
    return opt

def create_graph_lap(n):
    """
    Create graph laplacian of given size.
    """
    g = nx.random_geometric_graph(n, .5)
    #pdb.set_trace()
    #ensure connected!
    Lx = nx.laplacian_matrix(g, range(n))
    Lx = np.array(Lx.todense())
    Lx = np.array([[0, -.3, -.9],[-.3, 0, 0],[-.9, 0,0]])
    return Lx

def graph_to_lap(g):
    """
    Get Laplacian from nx graph g
    """
    if not isinstance(g, nx.Graph):
        g = nx.from_numpy_array(g)
    Lx = nx.laplacian_matrix(g).todense()  #args.Lx[torch.eye(len(args.Lx), dtype=torch.uint8)] = args.Lx.sum(0)    
    Lx = torch.from_numpy(Lx).to(dtype=torch_dtype)
    return Lx

def lap_to_graph(L):
    
    if isinstance(L, torch.Tensor):
        L = L.cpu().numpy()
    L = L.copy()
    np.fill_diagonal(L, 0)    
    return nx.from_numpy_array(-L)

def canonicalize_mx(mx):
    """
    Sort columns then rows to canonicalize mx. 
    Ensures symmetry.
    Square PSD matrix.
    """
    mx1 = mx.clone()
    diag = mx.diag()
    idx = diag.argsort(dim=0)
    n_mx = len(mx)
    for i in range(n_mx):
        mx[i] = mx1[idx[i]]
    mx1 = mx.clone()
    for i in range(n_mx):
        mx[:, i] = mx1[:, idx[i]]
    
    return mx 
    
def create_graph(n, gtype=None, seed=None, params={}):
    """
    Generate graph on n nodes of given type.
    """
    total_iter = 100
    cnt = 0
    while True:
        if gtype == 'block':
            
            if params['n_blocks'] == 3:
                m = n // 3
                g = nx.stochastic_block_model([m, m, n-2*m],[[0.98,0.01,.01],[0.01,0.98,.01],[0.01,.01,.98]], seed=seed)
            elif params['n_blocks'] == 4:
                m = n // 4
                g = nx.stochastic_block_model([m, m, m, n-3*m],[[.97,0.01,0.01,.01],[.01,0.97,0.01,.01],[.01,0.01,0.97,.01],[.01,0.01,0.01,.97] ], seed=seed)
            else:
                m = n // 2
                g = nx.stochastic_block_model([m, n-m],[[0.99,0.01],[0.01,0.99]], seed=seed)
        elif gtype == 'strogatz':
            g = nx.connected_watts_strogatz_graph(n, max(n//4, 3), p=.05, seed=seed)
        elif gtype == 'random_regular':
            #d = max(n//4, 2)  d*n must be even
            d = max(n//8, 2)
            if n*d % 2 == 1:
                n += 1
            g = nx.random_regular_graph(d, n, seed=seed)
        elif gtype == 'binomial':
            #also erdos renyi graph
            prob = params['prob']
            g = nx.binomial_graph(n, prob, seed=seed)
        elif gtype == 'barabasi':
            d = max(4, n//6)
            g = nx.barabasi_albert_graph(n, d, seed=seed)
        elif gtype == 'powerlaw_tree':            
            g = nx.random_powerlaw_tree(n, gamma=3, tries=1300, seed=seed)
        elif gtype == 'caveman':
            n_cliques = params['n_cliques']
            clique_sz = params['clique_sz']
            assert n_cliques * clique_sz == n
            g = nx.connected_caveman_graph(n_cliques, clique_sz)
        #elif gtype == 'binomial': 
        #    prob = params['prob']
        #    g = nx.binomial_graph(n, prob, seed=seed)
        elif gtype == 'random_geometric': 
            radius = params['radius']
            g = nx.random_geometric_graph(n, radius, seed=seed)
        elif gtype == 'barbell': 
            #note these are not numbers of nodes!
            g = nx.barbell_graph(n//2, 1)
        elif gtype == 'ladder': 
            g = nx.ladder_graph(n)            
        elif gtype == 'grid': 
            g = nx.grid_graph([n,n])            
        elif gtype == 'hypercube': 
            g = nx.hypercube_graph(n)            
        elif gtype == 'pappus': 
            g = nx.pappus_graph()            
        elif gtype == 'star': 
            g = nx.star_graph(n)   
        elif gtype == 'cycle': 
            g = nx.cycle_graph(n)   
        elif gtype == 'wheel': 
            g = nx.wheel_graph(n)   
        elif gtype == 'lollipop': 
            g = nx.lollipop_graph(n//2, 1)   
        else:
            raise Exception('graph type not supported ', gtype)

        remove_isolates(g)
        cnt += 1
        if nx.is_connected(g) or cnt > total_iter:
            if cnt > total_iter:
                g = g.subgraph(sorted(nx.connected_components(g), key=len)[-1]).copy()
            break
    return g

def fetch_data(dataset_name):
    #this assumes data lap file exists.
    #torch.save({'lap':lap_l, 'labels':node_labels, 'target':target}, '{}_lap.pt'.format(dataset_name))
    data = torch.load('{}_lap.pt'.format(dataset_name))
    return data['lap'], data['labels'], data['target']

def fetch_data_graphs(dataset_name):
    """
    #this assumes data (graph laplacians) are have been created or downloaded to file dataname_lap.pt.
    """
    #torch.save({'lap':lap_l, 'labels':node_labels, 'target':target}, '{}_lap.pt'.format(dataset_name))
    try:
        data = torch.load('data/{}_lap.pt'.format(dataset_name))
    except Exception:
        raise Exception('Dataset {} graph data not created yet. More data can be created using the generateData.py script as in README.'.format(dataset_name))
    graphs = []
    for g in data['lap']:
        graph = lap_to_graph(g)
        '''
        for i in range(len(g)):
            graph.add_node(i)
        pdb.set_trace()
        '''
        graphs.append(graph)    
            
    return graphs, data['labels'], np.array(data['target'])

def view_graph(L, soft_edge=False, labels=None, name=''):
    """
    Input:
    L: laplacian. Square Tensor (not upper triangular)
    labels: node labels.
    """
    plt.clf()
    if isinstance(L, torch.Tensor):        
        L = L.cpu().numpy()
    L = L.copy()
    #make diagonal 0?! negate?
    np.fill_diagonal(L, 0)
    L *= -1
    #pdb.set_trace()
    g = nx.from_numpy_array(L)
    fig = plt.figure()
    plt.axis('off')
    ax = plt.gca()
    #ax.axis('off')
    #layout = nx.kamada_kawai_layout(g)
    layout = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, layout, node_size=500, alpha=0.5, cmap=plt.cm.RdYlGn, node_color='r', ax=ax)
    if labels is None:        
        nx.draw_networkx_labels(g, layout, font_color='w', font_weight='bold', font_size=15, ax=ax)
    else:
        #labels can be determined from P
        nx.draw_networkx_labels(g, layout, labels=labels, font_color='k', font_size=12, ax=ax) #13
    if soft_edge:
        #sorting edges can be used to determine the cutoff
        #elarge = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] > 2]
        #esmall = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] <= 2 and d['weight'] > .99] #move to args!
        elarge = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] > .5] #.5
        esmall = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] <= .5 and d['weight'] > .19] #move to args! 3, 1.9 .2
        nx.draw_networkx_edges(g, layout, edgelist=elarge, width=3.5, ax=ax)
        nx.draw_networkx_edges(g, layout, edgelist=esmall, wifth=3, ax=ax)
    else:
        nx.draw_networkx_edges(g, layout, ax=ax)
    
    fig.savefig('data/view_graph_{}.jpg'.format(name))
    print('plot saved to {}'.format('data/view_graph_{}.jpg'.format(name)))
    plt.show()

def plot_confusion(tgt, pred, labels=None, name=''):
    """
    Input:
    tgt, pred: target and predicted classes.
    labels: node labels.
    """
    plt.clf()
    fig = plt.figure()
    #if isinstance(L, torch.Tensor):        
    #    L = L.cpu().numpy()
    ax = plt.gca()
    mx = sklearn.metrics.confusion_matrix(tgt, pred)
    #pdb.set_trace()
    img = plt.matshow(mx)
    path = 'data/confusion_mx_{}.jpg'.format(name)
    #plt.imsave(path, img)
    ax.legend()
    name2label = {'gw_cls':'GW', 'ot_cls':'COPT', 'combine_cls':'[COPT + GW]'}
    plt.title('{} Predictions'.format(name2label[name]), fontsize=20)
    plt.savefig(path)
    print('fig saved to ', path)


def plot_search_acc():
    '''
    Plot search acc. 
    '''
    plt.clf()
    x_l = [1, 3, 5, 10, 15]
    ot_acc = [0.9721962,0.9866296,0.9941481333,0.998,0.998]
    svd_acc = [0.814787,0.894257037,0.9349997,0.9774814073,0.9888888777]
    ot_std = [0.005516745247,0.003174584773,0.0002565744596,0.003464101615,0.003464101615] 
    svd_std = [0.01152346983,0.02409426815,0.02220130725,0.009109368462,0.009622514205] 

    fig = plt.figure()
    #ax = ax
    #plt.plot(x_l, ot_acc, '-o', label='COPT ')
    plt.errorbar(x_l, ot_acc, yerr=ot_std, marker='o', label='COPT sketches')
    #plt.plot(x_l, svd_acc, '-*', label='SVD  ')
    plt.errorbar(x_l, svd_acc, yerr=svd_std, marker='+', label='Spectral projections')
    
    plt.title('Classification acc of [COPT, GW] vs [spectral projections, GW] pipelines')
    plt.legend()
    plt.xlabel('Number of candidates allowed to 2nd stage')
    plt.ylabel('Classification accuracy')
    path = 'data/search_acc.jpg'#.format(name)    
    fig.savefig(path)
    print('fig saved to ', path)
    

    
def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
#create_dir(res_dir)

def normalizedMI(ar1, ar2):
    '''
    normalized mutual information of two input cluserings.
    Input: np arrays.
    '''
    score = sklearn.metrics.normalized_mutual_info_score(ar1, ar2)
    return score


def remove_edges(L, n_remove=1, seed=None):
    '''
    Adapted from GOT.
    '''
    rng = np.random.RandomState(seed)
    
    G = lap_to_graph(L)
    edges = np.triu(L, k=1).nonzero()
    
    removed = 0
    for idx in rng.permutation(edges[0].size):
        u, v = edges[0][idx], edges[1][idx]

        G.remove_edge(u,v)
        if nx.is_connected(G):
            removed += 1
        else:
            G.add_edge(u,v)
        if removed == n_remove:
            break
     
    return graph_to_lap(G) #nx.laplacian_matrix(G).todense()

def permute_nodes(l1, seed=None):
    '''
    Adapted from GOT.
    '''
    np.random.seed(seed)
    n = len(l1)
    idx = np.random.permutation(n)
    P_true = np.eye(n)
    P_true = P_true[idx]
    l2 = np.array(P_true @ l1 @ P_true.T)
    #pdb.set_trace()
    return np.double(l2), idx #P_true

def symmetrize(mx, inplace=True):
    """
    Make the matrix symmetric (according to upper right). In place.
    Input: torch tensor, square.
    """
    m, n = mx.size()
    assert m == n    
    mask = torch.ones(m, m)
    mask = torch.tril(mask, diagonal=-1)    
    #upper = torch.triu(mx).t()
    if not inplace:
        mx = mx.clone()
    mx[mask > 0] = torch.triu(mx, diagonal=1).t()[mask > 0]        
    return mx

def remove_isolates(g):
    """
    Remove isolated nodes from nx graph.
    """
    g.remove_nodes_from(list(nx.isolates(g)))

def load_data(fname):    
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        graphs = data['graphs']
        labels = data['labels']
    return graphs, labels

def read_lines(path):
    with open(path, 'r') as file:
        return file.readlines()

def parse_cls(st):
    ar = st.split('., ')
    return [int(i) for i in ar]
    
def plot_confusions():
    #plot_confusion(tgt, pred, labels=None, name=''):
    ot_cls = '0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 4., 1., 1., 4., 1., 1., 4., 1., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 9., 6., 9., 9., 9., 6., 6., 6., 9'
    gw_cls = '0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 6., 1., 1., 1., 1., 6., 6., 1., 6., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 6., 5., 6., 6., 6., 6., 5., 6., 6., 6., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9'
    combine_cls = '0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 4., 1., 1., 4., 1., 1., 1., 1., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 6., 5., 6., 6., 6., 6., 5., 6., 6., 6., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9'
    ot_cls = parse_cls(ot_cls)
    gw_cls = parse_cls(gw_cls)
    combine_cls = parse_cls(combine_cls)
    
    tgt_cls = []
    for i in [0, 1, 4, 5, 6, 9]:
        tgt_cls.extend([i]*10)
    names = ['ot_cls', 'gw_cls', 'combine_cls']
    for i, pred in enumerate([ot_cls, gw_cls, combine_cls]):
        plot_confusion(tgt_cls, pred, name=names[i])

def plot_convergence():
    '''
    plot Convergence
    '''
    conv = '99.77512221820409, 47.0570797352225, 45.454712183248795, 43.46666639511483, 40.9052209134619, 34.46097733377226, 24.431260224317242, 23.51762172579995, 22.485459983772515, 22.106446259828203, 21.534230884767375, 21.463743674621867'
    ar = conv.split(', ')
    conv_ar = [float(f) for f in ar]
    x_l = [20*i for i in list(range(len(conv_ar)))]
    fig = plt.figure()
    #plt.errorbar(x_l, ot_acc, yerr=ot_std, marker='o', label='COPT sketches')
    plt.plot(x_l, conv_ar, '-o', label='COPT distance')
    #plt.errorbar(x_l, svd_acc, yerr=svd_std, marker='+', label='Spectral projections')
    
    plt.title('COPT distance convergence sketching a 50-node graph to 15 nodes')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('COPT distance')
    path = 'data/convergence.jpg'#.format(name)    
    fig.savefig(path)
    print('fig saved to ', path)
    
if __name__ == '__main__':
    """
    For testing utils functions.
    """
    '''
    n = 2
    L = create_graph_lap(n)
    view_graph(L, soft_edge=True)
    '''
    #plot_search_acc()
    plot_cls_acc()
    #plot_convergence()
    #plot_confusions()
    
