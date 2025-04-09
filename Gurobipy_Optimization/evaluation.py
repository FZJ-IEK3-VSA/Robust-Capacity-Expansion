"""©Felix Engelhardt and Sebastian Kebrich (2024)
Contains files for plotting and evaluation of data."""

from colormap import rgb2hex
import matplotlib.pyplot as plt
import pandas as pd
from numpy import seterr


def plot_multiple_years(BaseTACs, reference_year="2018"):
    """Plots multiple years with total annual costs given in a dataframe index by commodities
    The reference column/year determines the hight of the red reference line"""

    commodities = BaseTACs.index
    print("Commodity name / Min / Max / Min Decrease in % / Max Increase in %")
    total_average = 0
    for commodity in commodities:
        base = BaseTACs.loc[[commodity]]
        total_average += (base.values).mean()
        seterr(invalid="ignore")
        print(
            commodity,
            (base.values).min(),
            (base.values).max(),
            (base.values).mean(),
            -(100 - 100 * (base.values).min() / (base.values).mean()),
            (100 * (base.values).max() / (base.values).mean()) - 100,
        )
        seterr(invalid="warn")
    print("Total average cost", total_average)

    columns_and_colors = {'Wind (onshore)':str(rgb2hex(17,146,251)),'Wind (offshore)':str(rgb2hex(109,38,142)),'Open field PV':str(rgb2hex(255,255,0)),
      'RoofTop PV':str(rgb2hex(255,255,171)),'Li-ion batteries':str(rgb2hex(35,104,120)),'Salt caverns (hydrogen)':str(rgb2hex(48,169,59)),
      'Electrolyzer':str(rgb2hex(51,81,149)),'CCGT hydrogen gas':str(rgb2hex(2,1,97)),'Electricity grid':str(rgb2hex(128,128,128)),'Hydrogen pipelines':str(rgb2hex(200,200,200))}

    names = {'Wind (onshore)':"Onshore wind",'Wind (offshore)':"Offshore wind",'Open field PV':'Open field PV',
      'RoofTop PV':'Rooftop PV','Li-ion batteries':'Li-ion batteries','Salt caverns (hydrogen)':'H$_2$ salt caverns',
      'Electrolyzer':'Electrolysers','CCGT hydrogen gas':'H$_2$ CCGT'}
    
    colours = [str(rgb2hex(17,146,251)),str(rgb2hex(109,38,142)),str(rgb2hex(255,255,0)),
      str(rgb2hex(255,255,171)),str(rgb2hex(35,104,120)),str(rgb2hex(48,169,59)),str(rgb2hex(51,81,149)),str(rgb2hex(2,1,97))]

    BaseTACs = BaseTACs.rename(index=names)

    plt.rcParams.update({"font.size": 20})
    columns_and_colors= [columns_and_colors[key] for key in commodities]

    BaseTACs = BaseTACs.copy()  # Ensure we don't modify the original DataFrame unexpectedly

    idx = list(BaseTACs.index)  # Get the index as a list
    idx[2], idx[3] = idx[3], idx[2]  # Swap the 3rd and 4th indices
    idx[5], idx[7] = idx[7], idx[5]
    BaseTACs = BaseTACs.loc[idx]  # Reorder the DataFrame based on the new index order

    print(BaseTACs.transpose().columns)
    ax = BaseTACs.transpose().plot(
        kind="bar", stacked=True, color=colours, figsize=(15, 8)
    )
    ax.set_ylabel("Total annual cost (TAC) [bn€/year]")
    ax.set_xlabel("Year")
    ax.axhline(BaseTACs.sum().mean(), color="red", linestyle="--", label="Average")
    ax.axhline(
        BaseTACs.sum()[reference_year], color="green", linestyle="--", label="2018"
    )
    for i, t in enumerate(ax.get_xticklabels()):
        if (i % 3) != 0:
            t.set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    plt.show()
    return True


def statistify(filepath_to_results: str):
    """Small script to return some statistical measures"""
    costs = []
    BaseTACs = pd.read_csv(filepath_to_results, index_col=0)
    print("BaseTACs:",BaseTACs, "\n")
    for counter in range(len(BaseTACs.columns)):
        final_system = BaseTACs.iloc[:, counter]
        costs.append(final_system.sum())


    for counter in range(len(BaseTACs.columns)):
        final_system = BaseTACs.iloc[:, counter]
        costs.append(final_system.sum())

    from numpy import argmin
    print(costs,argmin(costs))

    print("Costs min/max/mean")
    mean = sum(costs) / len(costs)
    print("Absolute", round(min(costs), 2), round(max(costs), 2), round(mean, 2))
    if mean != 0:
        print(
            "Relative to mean",
            round(min(costs) / mean - 1, 2),
            round(max(costs) / mean - 1, 2),
            "\n",
        )
    return BaseTACs

def compare(instance_filepath,reference_filepath="Weatherrobustness\\Results\\Base costs 1-node\\results_individual_years.csv"):
    from statistics import mean,stdev

    def get_baseline(filepath):
        """Small script to return mean and stdev of all installed supply capacities"""
        BaseTACs = pd.read_csv(filepath, index_col=0)
        means,stdevs = [],[]
        for row in range(len(BaseTACs.index)):
            rowlist = []
            for column in range(len(BaseTACs.columns)):
                rowlist.append(BaseTACs.iloc[row,column])
            means.append(mean(rowlist))
            stdevs.append(stdev(rowlist))
        return means,stdevs

    ref_means,ref_stdev = get_baseline(reference_filepath)
    inst_means,inst_stdev = get_baseline(instance_filepath)

    delta_means = [inst_means[i] - ref_means[i] for i in range(len(inst_means))]
    delta_stdev = [inst_stdev[i] - ref_stdev[i] for i in range(len(inst_stdev))]

    names = ['onshore','offshore','open field','rooftop','batteries','electrolyzer','CCGT','salt caverns']

    for i in range(len(names)):
        if ref_means[i] == 0:
            ref_means[i] = 1
            print("Warning: Reference mean was 0, setting to 1 to avoid division by zero")
        print(names[i],"Mean change:",round(delta_means[i],2),"Mean baseline:",round(ref_means[i],2),"Relative:",round(delta_means[i]/ref_means[i]*100,0))

    return True
#statistify("C:\\Users\\Felix Engelhardt\\Desktop\\robust-energy-systems\\Weatherrobustness\\Results\\Base costs 1-node\\results_individual_years.csv")

#compare("Weatherrobustness\\Results\\Alg8\\results_ALG8_critical.csv")
compare("Weatherrobustness\\Results\\Alg1\\results_ALG1_demand_side.csv")
print()
compare("Weatherrobustness\\Results\\Alg2\\results_ALG2_demand_side_smoothed.csv")
#statistify("Weatherrobustness\\Results\\Alg1\\results_ALG1_demand_side.csv")
BaseTACs = pd.read_csv("Weatherrobustness\\Results\\Alg8\\results_ALG8_critical.csv", index_col=0)
#BaseTACs = pd.read_csv("Weatherrobustness\\Results\\Base costs 1-node\\results_individual_years.csv", index_col=0)

plot_multiple_years(BaseTACs, reference_year="2018")