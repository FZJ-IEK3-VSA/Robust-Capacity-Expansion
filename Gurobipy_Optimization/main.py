"""©Felix Engelhardt and Sebastian Kebrich (2024)
Use to execute any of the algorithms."""

from evaluation import statistify, plot_multiple_years
from os import getcwd
import pandas as pd
from optimization import ALG4_load_stabilzation
from csv import writer

if __name__ == "__main__":
    # Parameters
    commodities = [
        "Wind (onshore)",
        "Wind (offshore)",
        "Open field PV",
        "RoofTop PV",
        "Li-ion batteries",
        "Electrolyzer",
        "CCGT hydrogen gas",
        "Salt caverns (hydrogen)",
    ]

    folderpath = getcwd()

    # Alg 4: Use ALG4_load_stabilization
    # Alg 6: Use Evaluate_load_stabilization with modifications=['dynamic']

    # Solve sample instance
    alltimes = []
    BaseTACs = pd.DataFrame(index=commodities)

    source_years = [1995]
    target_years = list(range(1995, 2020))
    algname = "None"

    for year in source_years:
        # solutions,times = evaluate_load_stabilzation([year],list(range(1980,2020)),folderpath=folderpath,commodities=commodities,modifications=['hydrogen demand'])
        solutions, times = ALG4_load_stabilzation(
            [year], target_years, folderpath=folderpath, commodities=commodities
        )
        alltimes.append(times)
        BaseTACs[str(year)] = solutions[-1]
        BaseTACs.to_csv(str(year) + "results_" + algname + ".csv")

        with open(str(year) + "results_" + algname + "_runtimes.csv", "w") as myfile:
            wr = writer(myfile)
            wr.writerow(alltimes)

    # Plot
    BaseTACs = statistify(
        "Weatherrobustness\\Results\\Alg1\\results_ALG1_demand_side.csv"
    )
    plot_multiple_years(BaseTACs, reference_year="2018")
