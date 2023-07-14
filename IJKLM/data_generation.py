import pandas as pd
import numpy as np
from collections import defaultdict


########## Data ##########
def create_fixed_data(m):
    J = [f"j{x}" for x in range(1, m + 1)]
    K = [f"k{x}" for x in range(1, m + 1)]
    L = [f"l{x}" for x in range(1, m + 1)]
    M = [f"m{x}" for x in range(1, m + 1)]

    jkl = pd.DataFrame(
        np.random.binomial(1, 0.05, size=(len(J) * len(K) * len(L))),
        index=pd.MultiIndex.from_product([J, K, L], names=["j", "k", "l"]),
        columns=["value"],
    ).reset_index()
    klm = pd.DataFrame(
        np.random.binomial(1, 0.05, size=(len(K) * len(L) * len(M))),
        index=pd.MultiIndex.from_product([K, L, M], names=["k", "l", "m"]),
        columns=["value"],
    ).reset_index()

    return J, K, L, M, jkl, klm


def create_variable_data(n, j, k):
    i = [f"i{x}" for x in range(1, n + 1)]

    ijk = pd.DataFrame(
        np.random.binomial(1, 0.05, size=(len(i) * len(j) * len(k))),
        index=pd.MultiIndex.from_product([i, j, k], names=["i", "j", "k"]),
        columns=["value"],
    ).reset_index()

    return i, ijk


def fixed_data_to_tuples(JKL, KLM):
    jkl = [
        tuple(x)
        for x in JKL.loc[JKL["value"] == 1][["j", "k", "l"]].to_dict("split")["data"]
    ]
    klm = [
        tuple(x)
        for x in KLM.loc[KLM["value"] == 1][["k", "l", "m"]].to_dict("split")["data"]
    ]
    return jkl, klm


def variable_data_to_tuples(IJK):
    ijk = [
        tuple(x)
        for x in IJK.loc[IJK["value"] == 1][["i", "j", "k"]].to_dict("split")["data"]
    ]
    return ijk


def fixed_data_to_dicts(JKL, KLM):
    JKL_dict = defaultdict(list)
    KLM_dict = defaultdict(list)
    for j, k, l in JKL:
        JKL_dict[j, k].append(l)
    for k, l, m in KLM:
        KLM_dict[k, l].append(m)
    return JKL_dict, KLM_dict

def str_to_num_idx(str_idx):
    i,j,k = str_idx
    i = int(''.join(c for c in i if c.isdigit()))
    j = int(''.join(c for c in j if c.isdigit()))
    k = int(''.join(c for c in k if c.isdigit()))   
    return i, j, k 

def variable_data_to_num_tuple(IJK):
    ijk = []
    for x in IJK.loc[IJK["value"] == 1][["i", "j", "k"]].to_dict("split")["data"]:
        ijk.append(tuple(str_to_num_idx(x)))
    return ijk    

def fixed_data_to_num_dicts(JKL,KLM):
    JKL_ndict = defaultdict(list)
    KLM_ndict = defaultdict(list)

    for jkl in JKL:
        j,k,l = str_to_num_idx(jkl)
        JKL_ndict[j,k].append(l)
    for klm in KLM:
        k,l,m = str_to_num_idx(klm)
        KLM_ndict[k,l].append(m)
    return JKL_ndict, KLM_ndict

def data_to_nnz_idx(I, IJK_numidx, JKL_ndict, KLM_ndict):
    # The indices will be converted to 0-based indexing.
    nnz_idx = {}
    for i in range(len(I)):
        jklm = []
        for _, j, k in list(filter(lambda x:x[0]==i+1, IJK_numidx)):        
            # 0-based indexing!!
            jklm += [(j-1, k-1, l-1, KLM_ndict[(k, l)][0]-1) for l in JKL_ndict[(j,k)] if (k,l) in KLM_ndict]
        if len(jklm)>0:
            nnz_idx[i] = jklm
    return nnz_idx