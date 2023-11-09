
import torch

class AngularAggLayer(nn.Module):
    def __init__(self, num_nodes, num_class, input_dim, labels, dropout):
        super(AngularAggLayer, self).__init__()
        self.register_buffer("mean_tensor", torch.randn(num_nodes, num_class))
        self.register_buffer("mag", torch.randn(1, input_dim))
        self.k = num_class
        self.N = num_nodes
        self.labels = labels
        self.dropout = dropout
        num = int(num_class*(num_class-1)/2)
        self.theta = nn.Parameter(torch.zeros(num,))
        
    def pre_norm(self, features):
        max_values, _ = torch.max(features, dim=0)
        theta = torch.arccos(features/max_values)
        new_fea = max_values * torch.cos(theta) + 1j*features*torch.sin(theta)
        return new_fea, max_values
    
    def post_norm(self, features):
        feature_norm = torch.abs(features)
        unit_norm_result = features / feature_norm
        normalized_result = unit_norm_result * self.mag
        return normalized_result
    
    # ! maybe trace the grad 
    def negative_symmetric_theta(self):
        matrix = torch.zeros(self.k, self.k)
        triu_indices = torch.triu_indices(self.k, self.k, offset=1)  # 上三角
        tril_indices = torch.tril_indices(self.k, self.k, offset=-1)  # 下三角
        matrix[triu_indices[0], triu_indices[1]] = self.theta
        matrix[tril_indices[1], tril_indices[0]] = -self.theta
        return matrix
    
    def get_centers(self, features, labels):
        one_hot = torch.nn.functional.one_hot(labels, num_classes=self.k)
        complex_one_hot = one_hot.t().to(torch.complex64)
        sum_by_label = torch.matmul(complex_one_hot, features)
        count_by_label = one_hot.sum(dim=0).unsqueeze(1)
        mean_tensor = sum_by_label / count_by_label
        return mean_tensor
    
    #! hard attention
    def get_label(self, feature, p = 1):
        dist = complex_dist(feature, self.mean_tensor, p = p)
        closest = torch.argmin(dist, dim=1)
        return closest
    
    def generate_matrix(self, w, label):
        label_indices = label.long()
        A_hat = w[label_indices.unsqueeze(1), label_indices.unsqueeze(0)]
        return A_hat

    def get_label_pro(self, feature, mean_tensor):
        '''
        soft-attention
        '''
        N = feature.shape[0]
        d = feature.shape[1] 
        k = mean_tensor.shape[0]

        fea_repeated_in_chunks = feature.repeat_interleave(N, dim=0)
        fea_repeated_alternating = feature.repeat(N, 1)
        all_combinations_matrix = torch.cat([fea_repeated_in_chunks, fea_repeated_alternating], dim=1)
        feature = all_combinations_matrix.view(N*N, 2*d)

        feature_1 = torch.matmul(feature ,torch.complex(self.w_real_1, self.w_imag_1)) #N*N,d
        feature_2 = torch.matmul(feature ,torch.complex(self.w_real_2, self.w_imag_2)) #N*N,d

        #  inner product
        first_part = torch.matmul(feature_1.unsqueeze(1), mean_tensor.unsqueeze(0).transpose(1, 2))
        second_part = torch.matmul(feature_2.unsqueeze(1), mean_tensor.unsqueeze(0).transpose(1, 2))

        first_inner_product = torch.real(first_part) + torch.imag(first_part)
        second_inner_product = torch.real(second_part) + torch.imag(second_part)

        i_index = torch.argmax(first_inner_product, dim=1)
        j_index = torch.argmax(second_inner_product, dim=1)

        return i_index.view(N,N), j_index.view(N,N)
    
    def generate_matrix_pro(self, X, Y, w):
        A_hat = w[X, Y]
        return A_hat
    
    def forward(self, x, A, l0=False):
        if l0:
            norm_features, self.mag = self.pre_norm(x)
        else:
            norm_features = x
            
        raw = norm_features
        self.mean_tensor = self.get_centers(norm_features, self.labels)
        
        #self.restr_theta()
        matrix = self.negative_symmetric_theta()
        
        fake_label = self.get_label(norm_features)
        A_hat = self.generate_matrix(matrix, fake_label)
        A = torch.where(A > 0, torch.tensor(1.0), A)
        A = torch.mul(A, A_hat)
        adj = torch.cos(A)+1j*torch.sin(A)
        message = torch.matmul(adj, norm_features)
        return self.post_norm(raw+message)
        

def complex_dist(c1,c2,p=1):
    real_part1 = c1.real
    imaginary_part1 = c1.imag
    real_part2 = c2.real
    imaginary_part2 = c2.imag
    
    real_distances = torch.cdist(real_part1, real_part2, p=p)
    imaginary_distances = torch.cdist(imaginary_part1, imaginary_part2, p =p)
    complex_distances = torch.sqrt(real_distances**2 + imaginary_distances**2)

    return complex_distances


class CGNN(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, labels, num_nodes):
        super(CGNN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(AngularAggLayer(num_nodes, nclass, nhidden, labels, dropout))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        
    def complex_relu(self, input):
        return F.relu(input.real).type(torch.complex64)+1j*F.relu(input.imag).type(torch.complex64)

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            if i ==0:
            #layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
                layer_inner = self.complex_relu(con(layer_inner,adj,True))
            else:
                layer_inner = self.complex_relu(con(layer_inner,adj))

        #layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        #--------#
        layer_inner = torch.angle(layer_inner)                   
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)
        

class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout,self).__init__()
        self.p = p
        
    def complex_dropout(self, input, p=0.5, training=True):
        # need to have the same dropout mask for real and imaginary part, 
        device = input.device
        mask = torch.ones(input.real.shape, dtype = torch.float32, device = device)
        mask = F.dropout(mask, p, training)*1/(1-p)
        mask.type(input.real.dtype)
        real,imag = mask * input.real,mask * input.imag
        return torch.complex(real,imag)

    def forward(self,input):
        if self.training:
            return self.complex_dropout(input,self.p)
        else:
            return input






def Augular_loss(y_true, theta, m, s):
    '''
    m: maybe try 0.5 first
    s: 1.0  if over-fit, reduce; vice-versa 
    '''
    # replace 0 => 1 and 1 => m in y_true
    M = (m - 1) * y_true + 1

    # add appropriate margin to theta
    new_theta = theta + M
    new_cos_theta = torch.cos(new_theta)

    # re-scale the cosines by a hyper-parameter s
    y_pred = s * new_cos_theta

    # the following part is the same as softmax loss
    numerators = torch.sum(y_true * torch.exp(y_pred), dim=1)
    denominators = torch.sum(torch.exp(y_pred), dim=1)
    loss = -torch.sum(torch.log(numerators / denominators))

    return loss
