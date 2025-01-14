import pandas as pd
import numpy as np

# import submodules
import IJKLM.data_generation as data
import visualization
from help import (
    create_directories,
    create_data_frame,
    incremental_range,
    save_to_json,
    below_time_limit,
    process_results,
    print_log_message,
    save_results,
)
#from IJKLM.run_gurobipy import run_gurobi, run_fast_gurobi
#from IJKLM.run_gams import data_to_gams, run_gams
from IJKLM.run_pyomo import run_pyomo, run_fast_pyomo
#from IJKLM.run_jump import run_julia

from IJKLM.run_mosek_fusion import run_mosek_fusion

############## Experiment ##########################
def run_experiment(
    cardinality_of_i, cardinality_of_j, solve, repeats, number, time_limit
):
    np.random.seed(13)

    # create empty frames for results
    df_fast_jump = create_data_frame()
    df_jump = create_data_frame()
    df_pyomo = create_data_frame()
    df_fast_pyomo = create_data_frame()
    df_gurobi = create_data_frame()
    df_fast_gurobi = create_data_frame()
    df_gams = create_data_frame()
    df_mosek_fusion = create_data_frame()

    # define the x axis
    N = list(incremental_range(1, cardinality_of_i + 1, 10, 10))

    # create fixed data and convert to tuples and dicts
    J, K, L, M, JKL, KLM = data.create_fixed_data(m=cardinality_of_j)
    jkl_tuple, klm_tuple = data.fixed_data_to_tuples(JKL, KLM)
    jkl_dict, klm_dict = data.fixed_data_to_dicts(jkl_tuple, klm_tuple)


    jkl_ndict, klm_ndict = data.fixed_data_to_num_dicts(jkl_tuple, klm_tuple)
    
    # save data to json for JuMP
    save_to_json(N, "N", "", "IJKLM")
    save_to_json(jkl_tuple, "JKL", "", "IJKLM")
    save_to_json(klm_tuple, "KLM", "", "IJKLM")

    # run experiment for every n in |I|
    for n in N:
        # create variable data and convert to tuples
        I, IJK = data.create_variable_data(n=n, j=J, k=K)
        ijk_num_tuple = data.variable_data_to_num_tuple(IJK)
        nnz_idx = data.data_to_nnz_idx(I, ijk_num_tuple, jkl_ndict, klm_ndict)
        ijk_tuple = data.variable_data_to_tuples(IJK)

        # save data to json for JuMP
        save_to_json(ijk_tuple, "IJK", f"_{n}", "IJKLM")
        
        # Gurobi
#        if below_time_limit(df_gurobi, time_limit):
#            rr = run_gurobi(I, ijk_tuple, jkl_tuple, klm_tuple, solve,
#                                    repeats=repeats, number=number)
#            df_gurobi = process_results(rr, df_gurobi)
#            print_log_message(language='GurobiPy', n=n, df=df_gurobi)

        # Fast Gurobi
#        if below_time_limit(df_fast_gurobi, time_limit):
#            rr = run_fast_gurobi(
#                I, ijk_tuple, jkl_tuple, klm_tuple, solve, repeats=repeats, number=number)
#            df_fast_gurobi = process_results(rr, df_fast_gurobi)
#            print_log_message(language='Fast GurobiPy', n=n, df=df_fast_gurobi)

        # GAMS
#        if below_time_limit(df_gams, time_limit):
#            data_to_gams(I, J, K, L, M, IJK, JKL, KLM)
#            rr = run_gams(solve, n, repeats=repeats, number=number)
#            df_gams = process_results(rr, df_gams)
#            print_log_message(language='GAMS', n=n, df=df_gams)

        # Pyomo
        if below_time_limit(df_pyomo, time_limit):
            rr = run_pyomo(
                I=I,
                IJK=ijk_tuple,
                JKL=jkl_tuple,
                KLM=klm_tuple,
                solve=solve,
                repeats=repeats,
                number=number,
            )
            df_pyomo = process_results(rr, df_pyomo)
            print_log_message(language="Pyomo", n=n, df=df_pyomo)

        # Fast Pyomo
        if below_time_limit(df_fast_pyomo, time_limit):
            rr = run_fast_pyomo(
                I=I,
                IJK=ijk_tuple,
                JKL=jkl_dict,
                KLM=klm_dict,
                solve=solve,
                repeats=repeats,
                number=number,
            )
            df_fast_pyomo = process_results(rr, df_fast_pyomo)
            print_log_message(language="Fast Pyomo", n=n, df=df_fast_pyomo)

        # MOSEK Fusion
        if below_time_limit(df_mosek_fusion, time_limit):
            rr = run_mosek_fusion(
                I=I,
                J=J, 
                K=K, 
                L=L, 
                M=M,
                nnz_idx=nnz_idx,
                solve=solve,
                repeats=repeats,
                number=number,
            )
            df_mosek_fusion = process_results(rr, df_mosek_fusion)
            print_log_message(language="MOSEK Fusion", n=n, df=df_mosek_fusion)

    # JuMP
    #df_fast_jump, df_jump = run_julia(solve, repeats, number, time_limit)

    # merge all results
    df = pd.concat(
        [
            #df_fast_jump,
            #df_jump,
            df_pyomo,
            df_fast_pyomo,
            #df_gurobi,
            #df_fast_gurobi,
            #df_gams,
            df_mosek_fusion
        ]
    ).reset_index(drop=True)

    # save results
    save_results(df, solve, "IJKLM")

    # plot results
    visualization.plot_results(df, cardinality_of_j, solve, "IJKLM")


if __name__ == "__main__":
    CI = 400000
    CJ = 20

    create_directories("IJKLM")
    solve = True
    # for solve in [False, True]:
    run_experiment(
        cardinality_of_i=CI,
        cardinality_of_j=CJ,
        solve=solve,
        repeats=3,
        number=1,
        time_limit=60,
    )
