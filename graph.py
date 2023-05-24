
"""
Class for core routine in coordinated optimal transport, can be applied to graph sketching, graph comparison, etc.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch_sparse
import utils_copt as utils
from torch.nn.parameter import Parameter
import sys
import time

import pdb

device = utils.device 

class GraphDist(nn.Module):
    # def __init__(self, Lx, m, n, args, Ly=None, take_ly_exp=True):
    def __init__(self, Lx, m, n, args, Ly=None, take_ly_exp=False):
        """
        Input: Lx and Ly are graph Laplacians, Lx being the input, origin graph,
        Ly the Laplacian of the target graph to be compared with.
        If sketching graph X, Ly should be None.
        Lx and Ly are *log* of upper triangular part of Laplacian of X and Y.
        """
        super(GraphDist, self).__init__()
        
        #P is m x n prob matrix, m is number of nodes in X, n is # of nodes in Y
        
        if args.fix_seed:
            torch.manual_seed(0)
        self.P = torch.empty(m, n).uniform_(1,2)
        
        #scale with sinkhorn iterations
        for _ in range(args.sinkhorn_iter):
            self.P /= (self.P.sum(1, keepdim=True)/n) #*m
            self.P /= (self.P.sum(0, keepdim=True)/m) #*n
        self.P = Parameter(self.P)
        #n x n mx, symmetric, off diag < 0, row & col sums are 0
        self.Lx_inv = mx_inv(Lx)
        
        #upper triangular part of Ly
        if Ly is None:
            if args.fix_seed:
                torch.manual_seed(0)        
            
            self.Ly = Parameter(torch.randn(n*(n-1)//2))            
            self.optim = optim.Adam([self.P, self.Ly], lr=.4)
            self.fix_ly = False
        else:
            Ly = Ly.clone()
            if take_ly_exp:
                assert len(Ly.shape) == 1
                self.Ly = realize_upper(Ly, args.n, take_ly_exp=take_ly_exp) #Parameter(Ly)
            else:
                self.Ly = Ly
            self.Ly = self.Ly.to(device=device)
            self.Ly_inv_rt, self.Ly_inv = mx_inv_sqrt(self.Ly)            
            
            self.optim = optim.Adam([self.P], lr=.35) #.4
            self.fix_ly = True
        
        #this is replaced by built-in scheduler in forward pass
        #milestones = [100*i for i in range(1, 4)] #[100, 200, 300]
        #self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=milestones, gamma=0.51)        
        
        self.Lx = Lx
        self.m, self.n = m, n
        
        self.args = args
        self.y_labels = None       
        
    def compute_graph_dist(self):
        Lx = self.Lx
        loss0 = sys.maxsize
        delta_l = []

        for i in range(self.args.n_epochs):
            loss, P, Ly = self.ot_dist(Lx, epoch=i)            
            self.optim.zero_grad()
            loss.backward()
            
            if self.args.verbose and i % 20 == 0:                
                cur_lr = self.optim.param_groups[0]['lr']
                sys.stdout.write('{} lr {}'.format(str(loss.cpu().item()), str(cur_lr)) + ', ')
            #torch.nn.utils.clip_grad_norm_(self.parameters(), 5.)
            self.optim.step()
            
            if (i % 101 == 0 and i < 2000):
                cur_lr = -1
                
                for p_group in self.optim.param_groups:                    
                    p_group['lr'] *= .7
                    cur_lr = p_group['lr']
                print_lr = False #True
                if print_lr:
                    print(i, cur_lr, i, loss.item())
            if self.args.lr_hike and loss0 - loss < .002 and i < self.args.n_epochs-200:            
                delta_l.append(loss)
                
                if len(delta_l) > self.args.hike_interval:
                    # can enable lr hiking to help escape local minima, described in supplement.
                    cur_lr = -1
                    for p_group in self.optim.param_groups:
                        p_group['lr'] = min(p_group['lr']*5, 4)
                        cur_lr = p_group['lr']
                        
                    #print('hike ', i)
                    #print(i, p_group['lr'], i, loss.item())
                    delta_l = []
                    if self.args.early_stopping:
                        break #can break here for early stopping
                    
            loss0 = loss
            
        '''
        avg_dur = total_dur / total_iter
        print('Avg iter timing {}'.format(avg_dur ))
        with open('time.txt', 'a') as f:
            f.write('{}\n'.format(avg_dur))
        ''' 
        P = P.clone().detach()
        if self.args.plot:
            labels = torch.topk(P, k=2, dim=0)[1].t().cpu().numpy()
            #self.y_labels = {k:str(label[0])+' '+str(label[1])+' '+str(label[2]) for k, label in enumerate(labels)}
            self.y_labels = {k:str(label[0])+' '+str(label[1]) for k, label in enumerate(labels)}
        
        return loss, P, Ly.clone().detach()
   
    def ot_dist(self, Lx, epoch=0):
        """
        Distance between two graphs. Evolve Ly and P simultaneouly
        1/|X| tr(L_X). self.Ly is the log of the actual laplacian.
        """
        if not self.fix_ly:
            ones = torch.ones(self.n, self.n, dtype=torch.uint8, device=device)
            Ly = torch.zeros(self.n, self.n, device=device)            
            Ly[torch.triu(ones, diagonal=1)] = -self.Ly**2            
            '''
            #can also use Huber function to enforce positivity.
            Ly_val = torch.abs(self.Ly.clone())
            Ly_val[Ly_val < 1] = -Ly_val[Ly_val < 1]**2/2
            Ly_val[torch.abs(self.Ly) >= 1] = -(torch.abs(self.Ly[torch.abs(self.Ly) >= 1])-.5)
            Ly[torch.triu(ones, diagonal=1)] = Ly_val
            '''            
            #Ly[torch.tril(ones, diagonal=-1)] = Ly[torch.triu(ones, diagonal=1)].t()
            #ensure laplacian
            Ly += Ly.clone().t()
            Ly[torch.eye(self.n, dtype=torch.uint8, device=device)] = -Ly.sum(0)

            Ly_inv_rt, Ly_inv = mx_inv_sqrt(Ly)
            #regularization
            #Ly += .1*torch.eye(self.n)
        else:
            Ly = self.Ly
            Ly_inv_rt, Ly_inv = self.Ly_inv_rt, self.Ly_inv
        
        '''
        Ly = utils.symmetrize(Ly, inplace=False)
        Ly_diag = Ly.diag()
        #Ly[(1-torch.eye(n)) > 0] = min(-Ly[(1-torch.eye(n)) > 0], Ly[(1-torch.eye(n)) > 0]) #off diag terms neg
        #Ly = torch.min(Ly, -Ly)
        Ly *= -1
        Ly[torch.eye(self.n) > 0] = Ly_diag
        '''
        #Ly[torch.eye(self.n) > 0] = 0   
        
        P = self.P.abs() 
        for _ in range(self.args.sinkhorn_iter):
            P = P / (P.sum(1, keepdim=True)/self.n) #*m
            P = P / (P.sum(0, keepdim=True)/self.m) #*n
        
        #approximate Ly^{-1/2}
        #when transport plan becomes uniform, mixed term goes to 0
        use_symeig = True
        if use_symeig:
            sqrt = torch.symeig(Ly_inv_rt @ P.t() @ self.Lx_inv @ P @ Ly_inv_rt, eigenvectors=True )
            #.clamp(min=0) here due to inconsistency between pytorch svd and symeig, ie svd gives PSD but symeig gives neg eval.
            ##self.w_dist = mx_tr(mx_inv(Lx))/self.m + mx_tr(mx_inv(Ly))/self.n - 2*torch.sqrt(sqrt[0].clamp(min=0)).sum() #
            self.w_dist = mx_tr(self.Lx_inv)*self.n + mx_tr(Ly_inv)*self.m - 2*torch.sqrt(sqrt[0].clamp(min=2e-20)).sum() #
        else:
            sqrt = Ly_inv_rt @ P.t() @ self.Lx_inv @ P @ Ly_inv_rt
            self.w_dist = mx_tr(self.Lx_inv)**2*self.n**2 + mx_tr(Ly_inv)**2*self.m**2 - 2*mx_tr(sqrt) 
        loss = self.w_dist
        
        #conditions on Ly, symmetric, off diag non-positive, row & col sums 0
        #row and col sums, can force to be 0
        ##ly_loss = torch.abs(Ly.sum(dim=1)).sum() + torch.abs(Ly.sum(dim=0)).sum() 
        #Ly_off_diag = Ly[1-torch.eye(self.n) > 0]
        #off diag should be non-positive
        #ly_loss += Ly_off_diag[Ly_off_diag > 0].sum()
        #ly_loss += torch.abs(torch.triu(Ly, diagonal=1) - torch.tril(Ly, diagonal=-1).t()).sum() #### #symmetry
        
        return loss.clamp(min=0), P, Ly

def realize_upper(upper, sz, take_ly_exp=False):
    ones = torch.ones(sz, sz, dtype=torch.uint8)
    Ly = torch.zeros(sz, sz)
    if take_ly_exp:
        Ly[torch.triu(ones, diagonal=1)] = -torch.exp(upper)
    else:
        Ly[torch.triu(ones, diagonal=1)] = upper        
    Ly += Ly.t()        
    Ly[torch.eye(sz, dtype=torch.uint8)] = -Ly.sum(0)
    return Ly

def mx_inv(mx):
    if isinstance(mx, torch_sparse.tensor.SparseTensor):
        mx = mx.to_dense()
    U, D, V = torch.svd(mx)
    eps = 0.009
    D_min = torch.min(D)
    if D_min < eps:
        D_1 = torch.zeros_like(D)
        D_1[D>D_min] = 1/D[D>D_min]
    else:
        D_1 = 1/D
    #D_1 = 1 / D #.clamp(min=0.005)
    
    return U @ D_1.diag() @ V.t()

def mx_inv_sqrt(mx):
    # singular values need to be distinct for backprop
    U, D, V = torch.svd(mx)
    D_min = torch.min(D)
    eps = 0.009
    if D_min < eps:
        D_1 = torch.zeros_like(D)        
        D_1[D>D_min] = 1/D[D>D_min] #.sqrt()
    else:
        D_1 = 1/D #.sqrt()
    #D_1 = 1 / D.clamp(min=0.005).sqrt()
    return U @ D_1.sqrt().diag() @ V.t(), U @ D_1.diag() @ V.t()

def mx_tr(mx):
    return mx.diag().sum()

def mx_svd(mx, topk):
    U, D, V = torch.svd(mx) 
    #topk x m
    evecs = U.t()[-topk-1:-1]
    row1 = evecs[-1, :]
    idx = torch.argsort(row1)
    evecs = evecs[:, idx]
    return evecs 

def graph_dist(args, plot=True, Ly=None, take_ly_exp=True):
    args.Lx = args.Lx.to(device)
    args.plot = plot
    model = GraphDist(args.Lx, args.m, args.n, args, Ly=Ly, take_ly_exp=take_ly_exp)
    model = model.to(device)
    loss, P, Ly = model.compute_graph_dist()
    Ly = Ly.cpu()
    #view graphs
    if plot:
        utils.view_graph(args.Lx, soft_edge=True, name='x')
        #Ly = realize_upper(model.Ly.detach(), args.n)
        utils.view_graph(Ly, soft_edge=True, name='y', labels=model.y_labels)
        pdb.set_trace()
    return loss, P, Ly

def visualize_graph(graph_type, args):
    g = utils.create_graph(args.m, graph_type)
    args.Lx = utils.graph_to_lap(g).to(device)
    args.m = len(args.Lx)
    
    model = GraphDist(args.Lx, args.m, args.n, args)
    model = model.to(device)
    loss, P, Ly = model.compute_graph_dist()
    Ly = Ly.cpu()
    #view graphs
    
    utils.view_graph(args.Lx, soft_edge=True, name='{}x'.format(graph_type))
    #Ly = realize_upper(model.Ly.detach(), args.n)
    utils.view_graph(Ly, soft_edge=True, name='{}y'.format(graph_type), labels=model.y_labels)
    pdb.set_trace()
    return loss, P, Ly
    
if __name__ == '__main__':
    args = utils.parse_args()
    args.m = 11 #5 6 5 12 (barbell)
    args.n = 9 #5 3 6 6
    if args.fix_seed:
        torch.manual_seed(0)
    test = torch.load('test.pt')
    args.Lx = test['q5']
    args.Ly = test['data19']
    args.m = len(args.Lx)
    args.n = len(args.Ly)
    graph_dist(args, plot=False, Ly=args.Ly, take_ly_exp=False)
    pdb.set_trace()
    args.Lx = torch.randn(args.m*(args.m-1)//2)  #torch.FloatTensor([[1, -1], [-1, 2]])
    args.Lx = realize_upper(args.Lx, args.m)
    args.n_epochs = 300 #1200 #100
    args.plot = False
    if False:#True: #False: #True:
        #graphs, labels = utils.load_data('data/graphs50.pkl')
        #args.Lx = utils.graph_to_lap(graphs[2])
        params = {'n_blocks':2}
        g2 = utils.create_graph(1000, gtype='block', params=params)
        args.Lx = utils.graph_to_lap(g2)
        args.m = len(args.Lx)
        '''
        params = {'n_blocks':3}
        g3 = utils.create_graph(50, gtype='block', params=params)        
        args.Ly = utils.graph_to_lap(g3)
        args.Ly = args.Lx.clone()
        args.n = len(args.Ly)        
        args.n = 15
        '''
        args.n = 200
        ##graph_dist(args, Ly=args.Ly, plot=False, take_ly_exp=False)
        #50 -> 7 => loss 12.3
        
        #pdb.set_trace()
    check_graph = False
    if check_graph:
        try:
            graphs, labels = utils.load_data('./data/graphs50.pkl')
        except FileNotFoundError:
            graphs, labels = utils.load_data('./copt/data/graphs.pkl')
        args.Lx = utils.graph_to_lap(graphs[2])
        ##g = utils.create_graph(12)
        ##args.Lx = utils.graph_to_lap(g)
        #args.Lx[torch.eye(len(args.Lx), dtype=torch.uint8)] = args.Lx.sum(0)
        args.m = args.n = len(args.Lx)  #for 30 to 10 the graphs can get to 9 loss for 280 epochs, .4 lr with schedule
        #50 -> 7 => loss 12.3
        args.n = 7
    vis = True #False #True
    if vis:
        graph_type = 'wheel' #'lollipop' #'barbell' # # #'wheel' #'cycle' #'ladder' #'hypercube' #'pappus' #'grid' #'hypercube'#'grid'#'ladder' #'barbell'
        visualize_graph(graph_type, args)
        
        pdb.set_trace()
    #args.Lx = torch.exp(torch.FloatTensor([[2, -2], [-2, 1]]))  #good initializations?! checks & stability
    print('args ', args)
    #graph_dist(args, plot=False, Ly=args.Lx, take_ly_exp=False)
    graph_dist(args, plot=False, take_ly_exp=False)
    

