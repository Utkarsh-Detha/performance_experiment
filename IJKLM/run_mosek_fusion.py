import timeit 
import logging
import pandas as pd
import numpy as np

import sys

import mosek.fusion as msk

########## MOSEK Fusion ##########

def run_mosek_fusion(I, J, K, L, M, IJK, JKL, KLM, solve, repeats, number):
    setup = {
        "I": I,
        "J": J,
        "K": K,
        "L": L,
        "M": M,
        "IJK": IJK,
        "JKL": JKL,
        "KLM": KLM,
        "solve": solve,
        "model_function": mosek_fusion,
    }

    r = timeit.repeat(
        "model_function(I, J, K, L, M, IJK, JKL, KLM, solve)",
        repeat=repeats,
        number=number,
        globals=setup,
    )


    result = pd.DataFrame(
        {
            "I": [len(I)],
            "Language": ["Pyomo"],
            "MinTime": [np.min(r)],
            "MeanTime": [np.mean(r)],
            "MedianTime": [np.median(r)],
        }
    )
    return result

def mosek_fusion(I, J, K, L, M, IJK, JKL, KLM, solve):
    model = msk.Model()

    model.objective(msk.ObjectiveSense.Minimize, 1.0)

    x = model.variable([len(I), len(J), len(K), len(L), len(M)], msk.Domain.greaterThan(0.0))

    sys.stdout.write(str([([([i, j, k, l, m]) for (_,l) in JKL[j] for (_,m) in KLM[k]]) for (i,j,k) in IJK]))

#    con_expr = msk.Expr.sum( [x.pick([[i, j, k, l, m] for (_,l) in JKL[j] for (_,m) in KLM[k]]) for (i,j,k) in IJK], 1)

#    model.constraint(con_expr, msk.Domain.greaterThan(0.0))

    if solve:
        model.setSolverParam("optimizerMaxTime", 0.0)
        model.setSolverParam("log", 0)
        model.solve()