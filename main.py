from tasks.task1_load_inspect import run_task1
from tasks.task2_missingness import run_task2

from tasks.task3_imputation import run_task3
import pandas as pd

from utils.config import PROCESSED_DIR
from tasks.task4_temporal import run_task4
from tasks.task5_spatial_network import run_task5
from tasks.task6_correlation_network import run_task6
#from tasks.task7_propagation import run_task7
from tasks.task8_parallelization import run_task8
from tasks.task9_forecasting import run_task9
from tasks.task10_final_visualization import run_task10



def main():
    print("MADRID AIR QUALITY PROJECT")
    
    # use sample for testing, switch to False for full dataset
    df = run_task1(use_sample=False,
        parallel=False,
        nrows=None,)
    df = run_task2(df)
    
    df = pd.read_parquet(PROCESSED_DIR / "cleaned_air_quality.parquet")

    df_imputed = run_task3(df)
    
    # main required analyses
    run_task4(df, magnitudes=["NO2", "NOX", "NO", "O3"])
    
    run_task5(
        df,
        thresholds_m=None,
        k_values=[2, 3, 4, 5],
        )
    
    run_task6(
    df,
    pollutants=["NO2", "NO", "NOX", "O3", "<PM10"],
    thresholds=[0.50, 0.60, 0.70, 0.80, 0.90],
    min_months=12,
    )

    # optional tasks
    #run_task7(df)
    

    # required performance analysis and final communication
    
    run_task8(threshold=0.6, 
              min_hours=24,
              n_workers=6,
              force_partitions=True,
              max_years= None,
              save_matrices=False,
    )
    
    run_task9(
        df,
        targets=["NO2", "NOX"],
        year_start=2019,     # where weather+traffic data becomes available
        max_sensors=15,      # biggest memory lever — limits the pivot width
        rf_n_estimators=100, # fewer trees = less RAM + faster
        rf_max_depth=6,
        )
    

    run_task10(df)
    

    print("\nProject finished. Check the outputs folder.")


if __name__ == "__main__":
    main()