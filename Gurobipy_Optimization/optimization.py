"""©Felix Engelhardt and Sebastian Kebrich (2024)
Includes algorithm to read in data, cluste in, merge reference years, and optimise ESMs
The latter refers to both single instances and feasibility testing algorithms"""

import pandas as pd
import gurobipy as gp
import numpy as np
from os.path import join
from sklearn.cluster import AgglomerativeClustering
from warnings import simplefilter
from gurobipy import GRB
from time import time


def read_in(folderpath: str, years: list):
    """Generic read in function, change folderpath to git as needed.
    All data is in GW / h / GWh if not otherwise specified."""

    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    simplefilter(action="ignore", category=FutureWarning)

    Eldata = pd.read_csv(
        join(
            folderpath,
            "InputData",
            "Demand",
            "El_demand_2050_DE_processed.csv",
        ),
        index_col=0,
    )  # Load one electricity demand time series for all reference years
    Eldata_normed = Eldata.sum(axis=1) / max(Eldata.sum(axis=1))

    df = pd.DataFrame()
    for year in years:  # PV/Wind data is different in each year
        Winddata = pd.read_csv(
            join(
                folderpath,
                "InputData",
                "WindOnshore",
                "Winddata" + str(year) + ".csv",
            ),
            index_col=0,
        )  # Load Wind ts
        WindOffdata = pd.read_csv(
            join(
                folderpath,
                "InputData",
                "WindOffshore",
                "Offshoredata" + str(year) + ".csv",
            ),
            index_col=0,
        )
        PVdata = pd.read_csv(
            join(
                folderpath,
                "InputData",
                "PV",
                "PVdata" + str(year) + ".csv",
            ),
            index_col=0,
        )  # Load PV ts

        # Remove non-float columns
        del Winddata["time"]
        del PVdata["time"]

        df["Wind" + str(year)] = (
            Winddata.sum(axis=1) / 38
        )  # Efficiency is normalised, since a single-node model is used
        df["WindOff" + str(year)] = WindOffdata.sum(axis=1) / 38
        df["PV" + str(year)] = PVdata.sum(axis=1) / 38
        df["El" + str(year)] = Eldata.sum(axis=1)
        df["Elnormed" + str(year)] = Eldata_normed

    capacities = {}
    capacities["Open field PV"] = (
        float(
            pd.read_csv(
                join(
                    folderpath,
                    "InputData",
                    "PV",
                    "MaxCap_NUTS2_OpenfieldPV.csv",
                ),
                index_col=0,
            ).sum(axis=1)
        )
        / 1000
    )  # Note that capacities are in MWh
    capacities["RoofTop PV"] = (
        float(
            pd.read_csv(
                join(
                    folderpath,
                    "InputData",
                    "PV",
                    "MaxCap_NUTS2_RoofTopPV.csv",
                ),
                index_col=0,
            ).sum(axis=1)
        )
        / 1000
    )
    capacities["Wind (onshore)"] = (
        float(
            pd.read_csv(
                join(
                    folderpath,
                    "InputData",
                    "WindOnshore",
                    "MaxCap_NUTS2_WindOnshore.csv",
                ),
                index_col=0,
            ).sum(axis=1)
        )
        / 1000
    )
    capacities["Wind (offshore)"] = (
        float(
            pd.read_csv(
                join(
                    folderpath,
                    "InputData",
                    "WindOffshore",
                    "MaxCap_NUTS2_WindOffshore.csv",
                ),
                index_col=0,
            ).sum(axis=1)
        )
        / 1000
    )
    capacities["Salt caverns"] = (
        float(
            pd.read_csv(
                join(
                    folderpath,
                    "InputData",
                    "Salt caverns",
                    "Hydrogencaverns_NUTS2_maxCap.csv",
                ),
                index_col=0,
            ).sum(axis=1)
        )
        / 1000
    )

    return df, capacities


def cluster_data(df: pd.DataFrame, n_clusters: int, years=list()):
    """For a time series given as dataframe, cluster it into n_clusters periods for one [year], or multiple years [year1,year2]."""
    clusters = []
    for year in years:
        conn = np.diag(np.ones((1, len(df.index) - 1))[0], -1) + np.diag(
            np.ones((1, len(df.index) - 1))[0], 1
        )
        conn[0][-1] = 1
        conn[-1][0] = 1
        _df = pd.DataFrame(
            df,
            columns=[
                "Wind" + str(year),
                "WindOff" + str(year),
                "PV" + str(year),
                "Elnormed" + str(year),
            ],
        )
        m = AgglomerativeClustering(
            n_clusters=n_clusters, distance_threshold=None, connectivity=conn
        )
        m.fit_predict(X=_df)
        df["cluster" + str(year)] = m.labels_

        for i in range(0, len(df.groupby("cluster" + str(year)))):
            cluster_columns = df.groupby("cluster" + str(year)).get_group(i)
            for cluster in cluster_columns:
                clusters.append(tuple(cluster_columns[cluster].index))
                break  # Slice out duplicate columns
    return clusters


def merge_years(df: pd.DataFrame, year1: int, year2: int):
    """Generates a new year from two existing years with double the length"""
    new_df = pd.DataFrame()
    new_df["Wind" + str(year1) + str(year2)] = pd.concat(
        [df["Wind" + str(year1)], df["Wind" + str(year2)]]
    )
    new_df["WindOff" + str(year1) + str(year2)] = pd.concat(
        [df["WindOff" + str(year1)], df["WindOff" + str(year2)]]
    )
    new_df["PV" + str(year1) + str(year2)] = pd.concat(
        [df["PV" + str(year1)], df["WindOff" + str(year2)]]
    )
    new_df["Elnormed" + str(year1) + str(year2)] = pd.concat(
        [df["Elnormed" + str(year1)], df["Elnormed" + str(year2)]]
    )
    new_df["El" + str(year1) + str(year2)] = pd.concat(
        [df["El" + str(year1)], df["El" + str(year2)]]
    )
    new_df.reset_index(drop=True, inplace=True)
    return new_df


def optimise_esm(
    df: pd.DataFrame,
    capacities,
    commodities,
    costs,
    year,
    cuts=[],
    assignments=False,
    x_eval=False,
    modifications=[],
    excess_load=None,
    target_years=[],
    printout=False,
):
    """Cost unit: 1e9 Euro, energy unit: GWh  Modification options:
    "darklull cut" => use darklull cut;
    "critical period cuts" => use cuts on critical periods;
    "global demand cuts" => use global demand cuts;
    "hydrogen demand" => model extra excess load as additional day;
    "extra load" => add load to designated time steps;
    "smoothed" => smooth adding of extra load"""

    # Step 0: Preprocessing / Fixed model parameters
    for (
        commodity
    ) in commodities:  # If not capacity is given, enable unlimited construction
        if commodity not in capacities:
            capacities[commodity] = GRB.INFINITY

    if (
        "hydrogen demand" in modifications
    ):  # Aggregate extra demand into additional hydrogen demand
        extra_load = sum(excess_load.values()) / 0.6  # Power return rate
    else:
        extra_load = 0

    # Step 1: Model
    m = gp.Model("ESM")
    m.setParam("OutputFlag", 0)  # Minimise Gurobi logging

    # Step 2: Variables
    investment, hydrogen_from_storage, hydrogen_to_storage, load_shedding = (
        {},
        {},
        {},
        {},
    )
    for commodity in (
        commodities
    ):  # These are the variables encoding investment decisions, i.e. building stuff
        # note that the round is there to avoid numerical troubels with UB/UL values
        investment[commodity] = m.addVar(
            lb=0,
            ub=round(capacities[commodity], 2),
            name=commodity,
            obj=costs[commodity],
        )

    stored_electrical_energy = m.addVars(
        df.index, lb=0, ub=GRB.INFINITY, name="Electricity storage level", obj=0
    )  # These variables encode stored energy
    stored_hydrogen = m.addVars(
        df.index, lb=0, ub=GRB.INFINITY, name="Hydrogen storage level", obj=0
    )
    stored_hydrogen[len(df.index)] = m.addVar(
        lb=0, ub=GRB.INFINITY, name="Hydrogen storage level", obj=0
    )  # For carry-over

    for timestep in df.index:
        hydrogen_from_storage[timestep] = m.addVar(
            lb=0, ub=GRB.INFINITY, name="Use hydrogen" + str(timestep), obj=0
        )  # This is for accounting only, to model energy storage/conversion
    for timestep in df.index:
        hydrogen_to_storage[timestep] = m.addVar(
            lb=0, ub=GRB.INFINITY, name="Store hydrogen" + str(timestep), obj=0
        )  # Note that we assume significant conversion losses for H2, but not for battery storage

    extra_energy_required = {}
    for cut in cuts:
        extra_energy_required[cut["id"]] = m.addVar(
            lb=0, ub=GRB.INFINITY, name="Extra energy" + str(cut["id"]), obj=0
        )

    if x_eval:
        load_shedding = m.addVars(
            df.index,
            lb=0,
            ub=capacities[commodity],
            obj=1000 * 50 * 1e-3,
        )  # This is a backup only, the price is arbitary

    m.update()

    # Step 3: Constraints

    for timestep in df.index:
        # Enforce storage capacity limits
        m.addConstr(
            stored_electrical_energy[timestep] <= investment["Li-ion batteries"]
        )
        m.addConstr(
            stored_hydrogen[timestep] <= investment["Salt caverns (hydrogen)"] / 0.7
        )

        # Bound gas conversion by investment
        m.addConstr(hydrogen_to_storage[timestep] <= investment["Electrolyzer"])
        m.addConstr(hydrogen_from_storage[timestep] <= investment["CCGT hydrogen gas"])

        # Model hydrogen conversion
        m.addConstr(
            stored_hydrogen[timestep + 1]
            == stored_hydrogen[timestep]
            - hydrogen_from_storage[timestep]
            + 0.7 * 0.6 * hydrogen_to_storage[timestep]
        )  # 0.7,0.6 are H2->electricity,electricity->H2 efficiencies

        # Enforce energy balance
        wind_energy = (
            df["Wind" + str(year)][timestep] * investment["Wind (onshore)"]
            + df["WindOff" + str(year)][timestep] * investment["Wind (offshore)"]
        )
        solar_energy = (
            df["PV" + str(year)][timestep] * investment["Open field PV"]
            + df["PV" + str(year)][timestep] * investment["RoofTop PV"]
        )
        gas_balance = hydrogen_from_storage[timestep] - hydrogen_to_storage[timestep]
        electricity_balance = (
            stored_electrical_energy[timestep]
            - stored_electrical_energy[max(timestep - 1, 0)]
        )

        demand = df["El" + str(year)][timestep]
        if "extra load" in modifications:
            demand += excess_load[timestep]

        if x_eval:
            m.addConstr(
                wind_energy
                + solar_energy
                + electricity_balance
                + gas_balance
                + load_shedding[timestep]
                >= demand
            )
        else:
            m.addConstr(
                wind_energy + solar_energy + electricity_balance + gas_balance >= demand
            )  # Only model slack if investments are already specified

    if assignments:  # Enforce existing assignments, if given
        for commodity in commodities:
            investment[commodity].lb = round(assignments[commodity], 6)
            if x_eval:
                investment[commodity].ub = round(assignments[commodity], 6)

    # Step 4: Add cuts, if necessary
    if not x_eval:
        for cut in cuts:
            # This ensures that enough energy can be produced
            m.addConstr(
                gp.quicksum(
                    df["Wind" + str(cut["year"])][timestep]
                    * investment["Wind (onshore)"]
                    + df["WindOff" + str(cut["year"])][timestep]
                    * investment["Wind (offshore)"]
                    + df["PV" + str(cut["year"])][timestep]
                    * investment["Open field PV"]
                    + df["PV" + str(cut["year"])][timestep] * investment["RoofTop PV"]
                    + investment["CCGT hydrogen gas"]
                    - df["El" + str(cut["year"])][timestep]
                    for timestep in cut["indices"]
                )
                >= cut["minimum"]
            )

            # This ensures that enough total energy is reserved
            m.addConstr(
                gp.quicksum(
                    -df["Wind" + str(cut["year"])][timestep]
                    * investment["Wind (onshore)"]
                    - df["WindOff" + str(cut["year"])][timestep]
                    * investment["Wind (offshore)"]
                    - df["PV" + str(cut["year"])][timestep]
                    * investment["Open field PV"]
                    - df["PV" + str(cut["year"])][timestep] * investment["RoofTop PV"]
                    + df["El" + str(cut["year"])][timestep]
                    for timestep in cut["indices"]
                )
                <= extra_energy_required[
                    cut["id"]
                ]  # We assume this energy has to be converted to H2 and back
            )

        # This ensures that enough energy is reserved early on
        for year in target_years:
            smaller_IDs = []
            cuts_in_year = [cut for cut in cuts if cut["year"] == year]

            for cut in cuts_in_year:
                smaller_IDs.append(cut["id"])
                m.addConstr(
                    0.7
                    * 0.6  # Discount for energy to H2 conversion
                    * gp.quicksum(
                        df["Wind" + str(cut["year"])][timestep]
                        * investment["Wind (onshore)"]
                        + df["WindOff" + str(cut["year"])][timestep]
                        * investment["Wind (offshore)"]
                        + df["PV" + str(cut["year"])][timestep]
                        * investment["Open field PV"]
                        + df["PV" + str(cut["year"])][timestep]
                        * investment["RoofTop PV"]
                        - df["El" + str(cut["year"])][timestep]
                        for timestep in range(0, cut["indices"][0])
                    )
                    + stored_hydrogen[min(df.index)]  # optional: Discount by conversion
                    >= gp.quicksum(extra_energy_required[ID] for ID in smaller_IDs)
                )

    # Enforce net zero energy across the year
    m.addConstr(
        stored_electrical_energy[min(df.index)]
        == stored_electrical_energy[max(df.index)]
    )

    if not x_eval:
        extra_load += gp.quicksum(extra_energy_required[cut["id"]] for cut in cuts)

    m.addConstr(
        stored_hydrogen[min(df.index)] + extra_load == stored_hydrogen[len(df.index)]
    )  # Note the slight difference due to the fact that the final step is required for hydrogen modelling

    # Step 5: Solution
    m.update()
    m.optimize()
    if printout:
        print()

    if m.Status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write("model.ilp")
        return False
    else:
        solution = []
        for commodity in commodities:
            solution.append(round(investment[commodity].x * costs[commodity], 3))
        if modifications != [] and printout:
            print("Results with cuts", year, "Total", m.getObjective().getValue())
        elif printout:
            print("Results without cuts", year, "Total", m.getObjective().getValue())
        if x_eval:  # TODO check assignments, check, return value: Can it be that no assignemnts are returned?
            excess_load = {}
            load_shedded = 0
            for timestep in df.index:
                excess_load[timestep] = load_shedding[timestep].x
                load_shedded += load_shedding[timestep].x
            if load_shedded > 0.01 and printout:
                print("Load shedding:", load_shedded)
            for commodity in commodities:
                assignments[commodity] = investment[commodity].x
            return (
                assignments,
                excess_load,
                load_shedded,
                solution,
                m.getObjective().getValue(),
            )
        else:
            assignments = {}
            for commodity in commodities:
                assignments[commodity] = round(investment[commodity].x, 6)
            return assignments, None, 0, solution, m.getObjective().getValue()


def ALG4_load_stabilzation(
    source_years: list, target_years: list, folderpath, commodities
):
    df, capacities = read_in(folderpath=folderpath, years=source_years + target_years)
    maxstep = 20

    costs = {
        "Wind (onshore)": 1 * (0.025 + 1.0 / 20),
        "Wind (offshore)": 2.99 * (0.025 + 1.0 / 20),
        "Open field PV": 0.32 * (0.017 + 1.0 / 20),
        "RoofTop PV": 0.474 * (0.001 + 1.0 / 20),
        "Li-ion batteries": 0.131 * (0.025 + 1.0 / 20),
        "Electrolyzer": 0.35 * (0.03 + 1.0 / 10),
        "CCGT hydrogen gas": 0.76 * (0.03 + 1.0 / 20),
        "Salt caverns (hydrogen)": 0.0007 * (0.02 + 1.0 / 40),
    }

    solutions, times = [], []

    starttime = time()
    years = set(source_years + target_years)
    clusters = {}
    for year in years:
        clusters[year] = sorted(cluster_data(df, 100, [year]), key=lambda x: x[0])

    for year in source_years:
        cuts = []

        iterationtime = time()

        print("\n######################################\nStabilising", year, "\n")
        total_excess_load = {index: 0 for index in df.index}
        fixed_assignment, dummy, dummy, solution, old_opt = optimise_esm(
            df,
            capacities,
            commodities,
            costs,
            year,
            cuts=cuts,
            modifications=[],
            target_years=target_years,
            excess_load=total_excess_load,
        )

        # Immideately add
        for otheryear in target_years:
            cuts.append(
                {"year": otheryear, "indices": df.index, "minimum": 0, "id": str(year)}
            )

        solutions.append(solution)
        for otheryear in target_years:
            load_shedding = 1
            old_load_shedding = None

            for step in range(maxstep):
                print("Year:", otheryear, "Iteration:", step, end=" ")

                # EVALUATE assignment
                dummy, excess_load, load_shedding, solution, opt = optimise_esm(
                    df,
                    capacities,
                    commodities,
                    costs,
                    otheryear,
                    assignments=fixed_assignment,
                    x_eval=True,
                )
                print(
                    "Original objective:",
                    old_opt,
                    "Objective:",
                    opt,
                    "Load shedding",
                    load_shedding,
                )

                # REOPTIMISE assignment
                if load_shedding > 0.01:
                    modifications = []

                    if step >= 1:
                        keydict = {}
                        counter = 0
                        for cut in cuts:
                            keydict[cut["id"]] = counter
                            counter += 1
                        counter = 0
                        for cluster in clusters[otheryear]:
                            if sum(excess_load[index] for index in cluster) > 0.01:
                                if str(year) + "-" + str(counter) in keydict:
                                    print(
                                        " Insufficient critical period cut on timesteps.",
                                        str(cluster[0]),
                                        str(cluster[-1]),
                                        " Year: "
                                        + str(otheryear)
                                        + " by: "
                                        + str(
                                            sum(excess_load[index] for index in cluster)
                                        )
                                        + " to "
                                        + str(
                                            cuts[
                                                keydict[str(year) + "-" + str(counter)]
                                            ]["minimum"]
                                        ),
                                    )

                                else:
                                    cuts.append(
                                        {
                                            "year": otheryear,
                                            "indices": cluster,
                                            "minimum": 0,
                                            "id": str(year) + "-" + str(counter),
                                        }
                                    )
                                    print(
                                        " New critical period cut on timesteps.",
                                        str(cluster[0]),
                                        str(cluster[-1]),
                                        " Year: "
                                        + str(otheryear)
                                        + " Extra Demand: "
                                        + str(
                                            sum(excess_load[index] for index in cluster)
                                        ),
                                    )
                            counter += 1
                        modifications.append("Critical period cuts")

                    if (
                        step >= 5 or old_load_shedding == load_shedding
                    ):  # Hardcoded to length 5 / triangular for smoothing time window
                        for index in df.index:
                            n = len(df.index)
                            total_excess_load[(index - 2) % n] += excess_load[index] / 9
                            total_excess_load[(index - 1) % n] += (
                                2 * excess_load[index] / 9
                            )
                            total_excess_load[index] += 3 * excess_load[index] / 9
                            total_excess_load[(index + 1) % n] += (
                                2 * excess_load[index] / 9
                            )
                            total_excess_load[(index + 2) % n] += excess_load[index] / 9
                        modifications.append("smoothed")
                        modifications.append("extra load")

                    fixed_assignment, dummy, dummy, solution, opt = optimise_esm(
                        df,
                        capacities,
                        commodities,
                        costs,
                        year,
                        cuts=cuts,
                        assignments=fixed_assignment,
                        modifications=modifications,
                        target_years=target_years,
                        excess_load=total_excess_load,
                    )
                solutions.append(solution)
                old_load_shedding = load_shedding

                if load_shedding <= 0.01:
                    break
        times.append(time() - iterationtime)
        print(
            "Total time for stabilization of", str(year), ":", (time() - iterationtime)
        )
    print("Total time for stabilization:", (time() - starttime))
    print("Total final cost:", opt, "compared to", old_opt)


def evaluate_load_stabilzation(
    source_years: list, target_years: list, folderpath, commodities, modifications=[]
):
    df, capacities = read_in(folderpath=folderpath, years=source_years + target_years)
    maxstep = 20

    costs = {
        "Wind (onshore)": 1 * (0.025 + 1.0 / 20),
        "Wind (offshore)": 2.99 * (0.025 + 1.0 / 20),
        "Open field PV": 0.32 * (0.017 + 1.0 / 20),
        "RoofTop PV": 0.474 * (0.001 + 1.0 / 20),
        "Li-ion batteries": 0.131 * (0.025 + 1.0 / 20),
        "Electrolyzer": 0.35 * (0.03 + 1.0 / 10),
        "CCGT hydrogen gas": 0.76 * (0.03 + 1.0 / 20),
        "Salt caverns (hydrogen)": 0.0007 * (0.02 + 1.0 / 40),
    }

    solutions, times = [], []

    starttime = time()
    years = set(source_years + target_years)
    if "critical period cuts" in modifications:
        clusters = {}
        for year in years:
            clusters[year] = sorted(cluster_data(df, 100, [year]), key=lambda x: x[0])

    for year in source_years:
        cuts = []

        iterationtime = time()

        print("\n######################################\nStabilising", year, "\n")
        total_excess_load = {index: 0 for index in df.index}
        fixed_assignment, dummy, dummy, solution, old_opt = optimise_esm(
            df,
            capacities,
            commodities,
            costs,
            year,
            cuts=cuts,
            modifications=modifications,
            target_years=target_years,
            excess_load=total_excess_load,
        )

        solutions.append(solution)
        for otheryear in target_years:
            load_shedding = 1

            if "global demand cuts" in modifications:
                for otheryear in target_years:
                    cuts.append(
                        {
                            "year": otheryear,
                            "indices": df.index,
                            "minimum": 0,
                            "id": str(year),
                        }
                    )

            for step in range(maxstep):
                print("Year:", otheryear, "Iteration:", step, end=" ")

                # EVALUATE assignment
                dummy, excess_load, load_shedding, solution, opt = optimise_esm(
                    df,
                    capacities,
                    commodities,
                    costs,
                    otheryear,
                    assignments=fixed_assignment,
                    x_eval=True,
                )
                print(
                    "Original objective:",
                    old_opt,
                    "Objective:",
                    opt,
                    "Load shedding",
                    load_shedding,
                )

                # PROCESS RESULTS
                if (
                    "smoothed" in modifications
                ):  # Hardcoded to length 5 / triangular for smoothing time window
                    for index in df.index:
                        n = len(df.index)
                        total_excess_load[(index - 2) % n] += excess_load[index] / 9
                        total_excess_load[(index - 1) % n] += 2 * excess_load[index] / 9
                        total_excess_load[index] += 3 * excess_load[index] / 9
                        total_excess_load[(index + 1) % n] += 2 * excess_load[index] / 9
                        total_excess_load[(index + 2) % n] += excess_load[index] / 9
                else:
                    total_excess_load = {
                        index: (total_excess_load[index] + excess_load[index])
                        for index in df.index
                    }

                # REOPTIMISE assignment
                if load_shedding > 0.01:
                    if "global demand cuts" in modifications:
                        for cut in cuts:
                            if cut["id"] == str(otheryear):
                                cut["minimum"] = sum(total_excess_load.values())

                    if "critical period cuts" in modifications:
                        keydict = {}
                        counter = 0
                        for cut in cuts:
                            keydict[cut["id"]] = counter
                            counter += 1

                        counter = 0

                        for cluster in clusters[otheryear]:
                            if sum(excess_load[index] for index in cluster) > 0.01:
                                if str(year) + "-" + str(counter) in keydict:
                                    print(
                                        " Insufficient critical period cut on timesteps.",
                                        str(cluster[0]),
                                        str(cluster[-1]),
                                        " Year: "
                                        + str(otheryear)
                                        + " by: "
                                        + str(
                                            sum(excess_load[index] for index in cluster)
                                        )
                                        + " to "
                                        + str(
                                            cuts[
                                                keydict[str(year) + "-" + str(counter)]
                                            ]["minimum"]
                                        ),
                                    )

                                else:
                                    cuts.append(
                                        {
                                            "year": otheryear,
                                            "indices": cluster,
                                            "minimum": 0,
                                            "id": str(year) + "-" + str(counter),
                                        }
                                    )
                                    print(
                                        " New critical period cut on timesteps.",
                                        str(cluster[0]),
                                        str(cluster[-1]),
                                        " Year: "
                                        + str(otheryear)
                                        + " Extra Demand: "
                                        + str(
                                            sum(excess_load[index] for index in cluster)
                                        ),
                                    )
                            counter += 1

                    fixed_assignment, dummy, dummy, solution, opt = optimise_esm(
                        df,
                        capacities,
                        commodities,
                        costs,
                        year,
                        cuts=cuts,
                        assignments=fixed_assignment,
                        modifications=modifications,
                        target_years=target_years,
                        excess_load=total_excess_load,
                    )
                solutions.append(solution)
                if load_shedding <= 0.01:
                    break
        times.append(time() - iterationtime)
        print(
            "Total time for stabilization of", str(year), ":", (time() - iterationtime)
        )
    print("Total time for stabilization:", (time() - starttime))
    print("Total final cost:", opt, "compared to", old_opt)
    return solutions, times
