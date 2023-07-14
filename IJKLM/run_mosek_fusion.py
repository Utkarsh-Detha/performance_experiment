import timeit 
import logging
import pandas as pd
import numpy as np
from collections import defaultdict

import sys

import mosek.fusion as msk

########## MOSEK Fusion ##########

def run_mosek_fusion(I, J, K, L, M, nnz_idx, solve, repeats, number):
    total_len = 0
    for i in nnz_idx.keys():
        total_len+=len(nnz_idx[i])

    setup = {
        "total_len":total_len,
        "nnz_idx":nnz_idx,
        "solve": solve,
        "model_function": mosek_fusion,
    }

    r = timeit.repeat(
        "model_function(total_len, nnz_idx, solve)",
        repeat=repeats,
        number=number,
        globals=setup,
    )


    result = pd.DataFrame(
        {
            "I": [len(I)],
            "Language": ["MOSEK Fusion"],
            "MinTime": [np.min(r)],
            "MeanTime": [np.mean(r)],
            "MedianTime": [np.median(r)],
        }
    )
    return result

def mosek_fusion(total_len, nnz_idx, solve):
    model = msk.Model()

    model.objective(msk.ObjectiveSense.Minimize, 1.0)

    x = model.variable(total_len, msk.Domain.greaterThan(0.0))

    c_e = []
    count = 0
    for i in nnz_idx.keys():
        size = len(nnz_idx[i])
        c_e.append(msk.Expr.sum(x.slice(count, count+size)))
        count += size
    c_e = msk.Expr.vstack(c_e)

    model.constraint(c_e, msk.Domain.greaterThan(0.0))

    if solve:
        model.setSolverParam("optimizerMaxTime", 0.0)
        model.setSolverParam("log", 0)
        model.solve()
        