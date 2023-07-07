import timeit 
import logging
import pandas as pd
import numpy as np

import mosek.fusion as msk

########## MOSEK Fusion ##########

def run_mosek_fusion(I, IJK, JKL, KLM, solve, repeats, number):
    setup = {
        "I": I,
        "IJK": IJK,
        "JKL": JKL,
        "KLM": KLM,
        "solve": solve,
        "model_function": mosek_fusion,
    }

    r = timeit.repeat(
        "model_function(I, IJK, JKL, KLM, solve)",
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


def mosek_fusion(I, IJK, JKL, KLM, solve):
    model = msk.Model()

    model.objective(msk.ObjectiveSense.Minimize, 1.0)

    x = model.variable()