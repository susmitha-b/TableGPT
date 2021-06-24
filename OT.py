import numpy as np
import torch
import torch.nn.functional as F

def cost_matrix(x, y):
    "Returns the matrix of $|x_i-y_j|^p$."
    "Returns the cosine distance"
    #NOTE: cosine distance and Euclidean distance
    # x_col = x.unsqueeze(1)
    # y_lin = y.unsqueeze(0)
    # c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    # return c
    x = F.normalize(x, p=2, dim=1, eps=1e-12)
    y = F.normalize(y, p=2, dim=1, eps=1e-12)
    tmp1 = torch.matmul(x, y.transpose(0, 1))
    cos_dis = 1 - tmp1

    x_col = torch.unsqueeze(x, 1)
    y_lin = torch.unsqueeze(y, 0)
    res = torch.sum(torch.abs(x_col - y_lin), 2)

    return cos_dis


def IPOT(C, n, m, beta=0.5):
    # sigma = tf.scalar_mul(1 / n, tf.ones([n, 1]))
    sigma = torch.ones(m, 1) / m.float()
    T = torch.ones([n, m])
    A = torch.exp(-C / beta)
    for t in range(50):
        Q = A * T
        for k in range(1):
            delta = 1 / (n.float() * torch.matmul(Q, sigma))
            sigma = 1 / (
                m.float() * torch.matmul(Q.transpose(0, 1), delta))
        # pdb.set_trace()
        tmp = torch.matmul(torch.diag(torch.squeeze(delta)), Q)
        T = torch.matmul(tmp, torch.diag(torch.squeeze(sigma)))
    
    return T

'''
def IPOT_np(C, beta=0.5):
    n, m = C.shape[0], C.shape[1]
    sigma = np.ones([m, 1]) / m
    T = np.ones([n, m])
    A = np.exp(-C / beta)
    for t in range(20):
        Q = np.multiply(A, T)
        for k in range(1):
            delta = 1 / (n * (Q @ sigma))
            sigma = 1 / (m * (Q.T @ delta))
        # pdb.set_trace()
        tmp = np.diag(np.squeeze(delta)) @ Q
        T = tmp @ np.diag(np.squeeze(sigma))
    return T
'''

def IPOT_distance(C, n, m):
    T = IPOT(C, n, m)
    distance = torch.trace(torch.matmul(C.transpose(0, 1), T))
    return distance

'''
def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)
    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret
'''

def IPOT_distance2(C, device, beta=1, t_steps=10, k_steps=1):
    # b, n, m = shape_list(C)
    b, n, m = C.size()
    sigma = (torch.ones(b, m, 1) / float(m)).to(device)  # [b, m, 1]
    T = torch.ones(b, n, m).to(device)
    A = torch.exp(-C / beta)  # [b, n, m]
    for t in range(t_steps):
        Q = A * T  # [b, n, m]
        for k in range(k_steps):
            delta = 1 / (float(n) * torch.matmul(Q, sigma))  # [b, n, 1]
            sigma = 1 / (float(m) * torch.matmul(Q.transpose(1, 2), delta))  # [b, m, 1]
        T = delta * Q * sigma.transpose(1, 2)  # [b, n, m]
    
    # distance = [torch.trace(d) for d in torch.matmul(C.transpose(1, 2), T)]
    distance = torch.matmul(C.transpose(1, 2), T).diagonal(dim1=-2, dim2=-1).sum(-1)
    
    return distance