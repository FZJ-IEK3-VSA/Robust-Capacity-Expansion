def get_csv(path: str):
    from os.path import isfile
    from pandas import read_csv

    if isfile(path):
        data = read_csv(path).round(3)
        data.index = data[data.columns[0]]  # Drop index columns
        del data[data.columns[0]]
        data = data.squeeze()
        data.sort_index(inplace=True)
        data.fillna(0, inplace=True)
        return data
    else:
        print("Import error for path ", path)
        return False


def pull_transmission_data(years: list):
    """Given a list of years, returns the maximum transmission capacity needed between regions across all years. Note the six hardcoded filepaths!"""
    if len(years) == 0:
        return False

    import FINE.IOManagement.xarrayIO as xrIO
    from FINE.IOManagement.xarrayIO import readNetCDFToDatasets
    from xarray import zeros_like
    from pandas import DataFrame
    from os import getcwd, path

    cwd = getcwd()
    DC_cable_data = readNetCDFToDatasets(
        path.join(cwd, "Results_38Regionen/NetCDF" + str(years[0]) + "-38Regionen.nc")
    )["Results"]["TransmissionModel"]["DC cables"].capacityVariablesOptimum
    H2_Network_data = readNetCDFToDatasets(
        path.join(cwd, "Results_38Regionen/NetCDF" + str(years[0]) + "-38Regionen.nc")
    )["Results"]["TransmissionModel"]["H2 pipelines"].capacityVariablesOptimum
    DC_TAC = zeros_like(DC_cable_data)
    H2_TAC = zeros_like(H2_Network_data)

    def update_data(old_data, new_data):
        if len(old_data) != len(new_data):
            return False
        for index1 in range(len(old_data)):
            for index2 in range(len(old_data)):
                old_data[index1, index2] = max(
                    old_data[index1, index2], new_data[index1, index2]
                )
        return old_data

    for year in years[0:]:
        this_years_DC = xrIO.readNetCDFToDatasets(
            path.join(cwd, "Results_38Regionen/NetCDF" + str(year) + "-38Regionen.nc")
        )["Results"]["TransmissionModel"]["DC cables"].capacityVariablesOptimum
        this_years_H2 = xrIO.readNetCDFToDatasets(
            path.join(cwd, "Results_38Regionen/NetCDF" + str(year) + "-38Regionen.nc")
        )["Results"]["TransmissionModel"]["H2 pipelines"].capacityVariablesOptimum
        this_years_DC_TAC = xrIO.readNetCDFToDatasets(
            path.join(cwd, "Results_38Regionen/NetCDF" + str(year) + "-38Regionen.nc")
        )["Results"]["TransmissionModel"]["DC cables"].TAC
        this_years_H2_TAC = xrIO.readNetCDFToDatasets(
            path.join(cwd, "Results_38Regionen/NetCDF" + str(year) + "-38Regionen.nc")
        )["Results"]["TransmissionModel"]["H2 pipelines"].TAC
        DC_cable_data = update_data(DC_cable_data, this_years_DC)
        H2_Network_data = update_data(H2_Network_data, this_years_H2)
        DC_TAC = update_data(DC_TAC, this_years_DC_TAC)
        H2_TAC = update_data(H2_TAC, this_years_H2_TAC)

    print(
        "Determined maximal transmission grid. TAC:",
        int(100 * (sum(sum(DC_TAC)) + sum(sum(H2_TAC)))) / 200,
    )

    region_index = (
        DC_cable_data.space
    )  # Write all to csv, but ensure indexing is preserved
    DC_cable_data = DataFrame(
        DC_cable_data.values, index=region_index, columns=region_index
    )
    region_index = H2_Network_data.space
    H2_Network_data = DataFrame(
        H2_Network_data.values, index=region_index, columns=region_index
    )
    DC_cable_data.to_csv(path.join(cwd, "InputData", "Grid", "DC_grid_data.csv"))
    H2_Network_data.to_csv(path.join(cwd, "InputData", "Grid", "H2_grid_data.csv"))

    return DC_cable_data, H2_Network_data


def pull_construction_data(years: list, commodities: list):
    """Given a (list of) year(s), returns the maximum capacity installed per region. Note the six hardcoded filepaths!"""
    if len(years) == 0:
        return False

    import FINE.IOManagement.xarrayIO as xrIO
    from FINE.IOManagement.xarrayIO import readNetCDFToDatasets
    from xarray import zeros_like
    from pandas import DataFrame
    from os import getcwd, path

    typeoftec = {
        "Wind (onshore)": "SourceSinkModel",
        "Wind (offshore)": "SourceSinkModel",
        "Open field PV": "SourceSinkModel",
        "RoofTop PV": "SourceSinkModel",
        "Electrolyzer": "ConversionModel",
        "CCGT hydrogen gas": "ConversionModel",
        "Salt caverns (hydrogen)": "StorageModel",
        "Li-ion batteries": "StorageModel",
    }

    cwd = getcwd()
    data = readNetCDFToDatasets(
        path.join(cwd, "Results_38Regionen/NetCDF" + str(years[0]) + "-38Regionen.nc")
    )["Results"]

    capacities, TAC = {}, {}
    for commodity in commodities:
        capacities[commodity] = data[typeoftec[commodity]][
            commodity
        ].capacityVariablesOptimum
        TAC[commodity] = zeros_like(capacities[commodity])

    def update_data(old_data, new_data):
        if len(old_data) != len(new_data):
            return False
        for index in range(len(old_data)):
            old_data[index] = max(old_data[index], new_data[index])
        return old_data

    for commodity in commodities:
        for year in years[0:]:
            this_years_capacities = xrIO.readNetCDFToDatasets(
                path.join(
                    cwd, "Results_38Regionen/NetCDF" + str(year) + "-38Regionen.nc"
                )
            )["Results"][typeoftec[commodity]][commodity].capacityVariablesOptimum
            this_years_TAC = xrIO.readNetCDFToDatasets(
                path.join(
                    cwd, "Results_38Regionen/NetCDF" + str(year) + "-38Regionen.nc"
                )
            )["Results"][typeoftec[commodity]][commodity].TAC
            capacities[commodity] = update_data(
                capacities[commodity], this_years_capacities
            )
            TAC[commodity] = update_data(TAC[commodity], this_years_TAC)
        print(
            "Determined maximal " + commodity + " TAC:",
            int(1000 * sum(TAC[commodity])) / 1000,
        )
        region_index = capacities[commodity].space
        capacities[commodity] = DataFrame(capacities[commodity], index=region_index)
        if len(years) == 1:
            capacities[commodity].to_csv(
                path.join(
                    cwd,
                    "InputData",
                    "CapacityFix",
                    str(years[0]) + "_" + str(commodity) + ".csv",
                )
            )
        else:
            capacities[commodity].to_csv(
                path.join(
                    cwd,
                    "InputData",
                    "CapacityFix",
                    str(years[0]) + "_" + str(years[1]) + "_" + str(commodity) + ".csv",
                )
            )

    return capacities, TAC


def SimulateYear(
    year: int,
    min_cap_year: int = None,
    update: bool = True,
    use_hydrogen: bool = True,
    fix_assignments: bool = None,
):
    import fine as fn
    import pandas as pd
    import os
    import fine.IOManagement.xarrayIO as xrIO
    
    print("\n Start SimulateYear. \n")

    ############################# Part 1: Import the Data #############################

    cwd = os.getcwd()

    # 1.1. Import PV
    
    PVdata = pd.read_csv(os.path.join(cwd, "InputData", "PV", f"PVdata{year}.csv"), index_col=0)

    PVmaxCapOpenfieldPV = pd.read_csv(
        os.path.join(cwd, "InputData", "PV", "MaxCap_NUTS2_OpenfieldPV.csv"), index_col=0
    )
    PVmaxCapRoofTopPV = pd.read_csv(
        os.path.join(cwd, "InputData", "PV", "MaxCap_NUTS2_RoofTopPV.csv"), index_col=0
    )

    if update:
        print("PV data read in.")

    # 1.2 Import Wind
    Winddata = pd.read_csv(
        os.path.join(cwd, "InputData", "WindOnshore", f"Winddata{year}.csv"), index_col=0
    )

    WinddataOffshore = pd.read_csv(
        os.path.join(cwd, "InputData", "WindOffshore", f"Offshoredata{year}.csv"), index_col=0
    )

    WindOnshoremaxCap = pd.read_csv(
        os.path.join(cwd, "InputData", "WindOnshore", "MaxCap_NUTS2_WindOnshore.csv"), index_col=0
    )
    WindOffshoremaxCap = pd.read_csv(
        os.path.join(cwd, "InputData", "WindOffshore", "MaxCap_NUTS2_WindOffshore.csv"),index_col=0
    )
    
    if update:
        print("Wind data read in.")

    # 1.3 Import Electricity Demand
    Electricity_demand = pd.read_csv(
        os.path.join(cwd, "InputData", "Demand", "El_demand_2050_DE_processed.csv"), index_col=0
    )

    if update:
        print("Electricity data read in.")

    # 1.4 Import Hydrogen (optional)
    if use_hydrogen:
        H2cavernmaxCap = pd.read_csv(
            os.path.join(
                cwd, "InputData", "Salt caverns", "Hydrogencaverns_NUTS2_maxCap.csv"
            )
        )
        H2cavernmaxCap = H2cavernmaxCap.transpose().round(3)
        if update:
            print("Hydrogen storage data read in.")

    distances = pd.read_csv(
            os.path.join(
                cwd, "InputData", "Distances", "Distances_NUTS2_DE.csv"
            ),index_col=0)

    print("All data loaded.\n")

    # 1.6 Set Locations
    locations = {
        "DEF0",
        "DE60",
        "DE91",
        "DE92",
        "DE93",
        "DE94",
        "DE50",
        "DEA1",
        "DEA2",
        "DEA3",
        "DEA4",
        "DEA5",
        "DE71",
        "DE72",
        "DE73",
        "DEB1",
        "DEB2",
        "DEB3",
        "DE11",
        "DE12",
        "DE13",
        "DE14",
        "DE21",
        "DE22",
        "DE23",
        "DE24",
        "DE25",
        "DE26",
        "DE27",
        "DEC0",
        "DE30",
        "DE40",
        "DE80",
        "DED4",
        "DED2",
        "DED5",
        "DEE0",
        "DEG0",
    }

    # 1.7 Set Commodities
    if use_hydrogen:
        commodityUnitDict = {
            "electricity": r"GW$_{el}$",
            "hydrogen_gas": r"GW$_{H_{2},LHV}$",
        }
        commodities = {"electricity", "hydrogen_gas"}
    else:
        commodityUnitDict = {"electricity": r"GW$_{el}$"}
        commodities = {"electricity"}

    # 1.8 Set Number of Timesteps
    numberOfTimeSteps = 8760

    # 1.9 Fix Assignments if given
    if not fix_assignments:
        fixed_onshore_capacity = None
        fixed_offshore_capacity = None
        fixed_openfield_capacity = None
        fixed_rooftop_capacity = None
        fixed_electrolyzer_capacity = None
        fixed_CCGT_capacity = None
        fixed_cavern_capacity = None
        fixed_LI_capacity = None
    else:
        if not min_cap_year:
            min_cap_year = year
        fixed_onshore_capacity = get_csv(
            os.path.join(
                cwd,
                "InputData",
                "CapacityFix",
                str(min_cap_year) + "_Wind (onshore).csv",
            )
        )
        fixed_offshore_capacity = get_csv(
            os.path.join(
                cwd,
                "InputData",
                "CapacityFix",
                str(min_cap_year) + "_Wind (offshore).csv",
            )
        )
        fixed_openfield_capacity = get_csv(
            os.path.join(
                cwd,
                "InputData",
                "CapacityFix",
                str(min_cap_year) + "_Open field PV.csv",
            )
        )
        fixed_rooftop_capacity = get_csv(
            os.path.join(
                cwd, "InputData", "CapacityFix", str(min_cap_year) + "_RoofTop PV.csv"
            )
        )
        fixed_electrolyzer_capacity = get_csv(
            os.path.join(
                cwd, "InputData", "CapacityFix", str(min_cap_year) + "_Electrolyzer.csv"
            )
        )
        fixed_CCGT_capacity = get_csv(
            os.path.join(
                cwd,
                "InputData",
                "CapacityFix",
                str(min_cap_year) + "_CCGT hydrogen gas.csv",
            )
        )
        fixed_cavern_capacity = get_csv(
            os.path.join(
                cwd,
                "InputData",
                "CapacityFix",
                str(min_cap_year) + "_Salt caverns (hydrogen).csv",
            )
        )
        fixed_LI_capacity = get_csv(
            os.path.join(
                cwd,
                "InputData",
                "CapacityFix",
                str(min_cap_year) + "_Li-ion batteries.csv",
            )
        )

        PVmaxCapOpenfieldPV["Cap_max_GW"] = fixed_openfield_capacity
        PVmaxCapRoofTopPV["Cap_max_GW"] = fixed_rooftop_capacity
        WindOffshoremaxCap["Cap_max_GW"] = fixed_offshore_capacity
        WindOnshoremaxCap["Cap_max_GW"] = fixed_onshore_capacity

    ############################# Generate the Model #############################

    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=0,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Slack",
            commodity="electricity",
            hasCapacityVariable=False,
            commodityCost=0.0005
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Wind (onshore)",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=Winddata,
            capacityMax=WindOnshoremaxCap['Cap_max_GW'],
            investPerCapacity=1,
            opexPerCapacity=0.025,
            interestRate=0.08,
            economicLifetime=20,
            capacityMin=fixed_onshore_capacity,
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Wind (offshore)",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=WinddataOffshore,
            capacityMax=WindOffshoremaxCap["Cap_max_GW"],
            investPerCapacity=2.99,
            opexPerCapacity=0.75, 
            interestRate=0.08,
            economicLifetime=20,
            capacityMin=fixed_offshore_capacity,
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Open field PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=PVdata,
            capacityMax=PVmaxCapOpenfieldPV["Cap_max_GW"],
            investPerCapacity=0.32,
            opexPerCapacity=0.32 * 0.017,
            interestRate=0.08,
            economicLifetime=20,
            capacityMin=fixed_openfield_capacity,
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="RoofTop PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=PVdata,
            capacityMax=PVmaxCapRoofTopPV["Cap_max_GW"],
            investPerCapacity=0.474,
            opexPerCapacity=0.01, 
            interestRate=0.04,
            economicLifetime=20,
            capacityMin=fixed_rooftop_capacity,
        )
    )

    if use_hydrogen:
        esM.add(
            fn.Conversion(
                esM=esM,
                name="Electrolyzer",
                physicalUnit=r"GW$_{el}$",
                commodityConversionFactors={"electricity": -1, "hydrogen_gas": 0.7},
                hasCapacityVariable=True,
                investPerCapacity=0.350,
                opexPerCapacity=0.350 * 0.03,
                interestRate=0.08,
                economicLifetime=10,
                capacityMin=fixed_electrolyzer_capacity,
            )
        )

        esM.add(
            fn.Conversion(
                esM=esM,
                name="CCGT hydrogen gas",
                physicalUnit=r"GW$_{el}$",
                commodityConversionFactors={
                    "electricity": 1,
                    "hydrogen_gas": -1.6667,
                }, 
                investPerCapacity=0.76,
                opexPerCapacity=0.76 * 0.03,
                interestRate=0.08,
                economicLifetime=20,
                capacityMin=fixed_CCGT_capacity,
            )
        )

        esM.add(
            fn.Storage(
                esM=esM,
                name="Salt caverns (hydrogen)",
                commodity="hydrogen_gas",
                hasCapacityVariable=True,
                chargeEfficiency=0.98,
                dischargeEfficiency=0.99,
                cyclicLifetime=10000,
                selfDischarge=0,
                chargeRate=1,
                dischargeRate=1,
                doPreciseTsaModeling=False,
                capacityMax=H2cavernmaxCap[0],
                investPerCapacity=0.0007,
                opexPerCapacity=0.0007 * 0.02,
                interestRate=0.08,
                economicLifetime=40,
                capacityMin=fixed_cavern_capacity,
            )
        )

    esM.add(
        fn.Storage(
            esM=esM,
            name="Li-ion batteries",
            commodity="electricity",
            hasCapacityVariable=True,
            chargeEfficiency=0.96,
            cyclicLifetime=10000,
            dischargeEfficiency=0.96,
            selfDischarge=0.0000423,
            chargeRate=1,
            dischargeRate=1,
            doPreciseTsaModeling=False,
            investPerCapacity=0.131,
            opexPerCapacity=0.00328,
            interestRate=0.08,
            economicLifetime=15,
            capacityMin=fixed_LI_capacity,
        )
    )


    esM.add(
        fn.Transmission(
            esM=esM,
            name="DC cables",
            commodity="electricity",
            distances=distances,
            losses=0.000035,
            investPerCapacity=0.000860,
            opexPerCapacity=0.00003,
            interestRate=0.08,
            economicLifetime=40
        )
    )


    if use_hydrogen:
        esM.add(
            fn.Transmission(
                esM=esM,
                name="H2 pipelines",
                commodity="hydrogen_gas",
                distances=distances,
                losses=0,  # 1/km,
                investPerCapacity=185e-6,
                opexPerCapacity=0.000001,
                economicLifetime = 40,
                interestRate=0.08,
            )
        )

    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=Electricity_demand,
        )
    )

    if update:
        print("Model build.")

    ############################# 3. Solve the Model #############################

    # 3.1 Solve the MIP
    esM.optimize(
        timeSeriesAggregation=False,
        optimizationSpecs="Method=2 BarConvTol=1e-5 Presolve=2 AggFill=1 Crossover=0 BarHomogeneous=1",
    )

    ############################# Return Results #############################

    _ = xrIO.writeEnergySystemModelToNetCDF(
        esM,
        outputFilePath=os.path.join(cwd, "Results", f"NetCDF_new_{year}.nc"),
        overwriteExisting=True,
    )
    print(f"Done with {year}")


def TestFeasibility(year1: int, year2: int, region_resolution: int):
    if year1 not in range(1980, 2020) or year2 not in range(1980, 2020):
        return False
    print("\nApply data from", year2, "as basis for", year1)
    SimulateYear(
        year1,
        min_cap_year=year2,
        fix_assignments=True,
        number_of_regions=region_resolution,
    )


def CriticalTimePeriod(years: list, clusters: list, referenceyears: list, Gap: int):
    import pandas as pd
    import os
    from os import path
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from fine.IOManagement.xarrayIO import readNetCDFToDatasets
    from xarray import zeros_like



    cwd = os.getcwd()
    Operational_data = pd.DataFrame()

    #Load, normalize and aggregate the electricity data
    Electricity_data = pd.read_csv(os.path.join(cwd, "InputData", "Demand", "El_demand_2050_DE_processed.csv"), index_col=0)
    Electricity_data_aggregated_normalized = Electricity_data.sum(axis=1) / max(Electricity_data.sum(axis=1))
    Operational_data["Electricity demand normalized"] = Electricity_data_aggregated_normalized
    Operational_data["Electricity demand"] = Electricity_data.sum(axis=1)
                     
    for year in years:
        WindOndata = pd.read_csv(
            os.path.join(cwd, "InputData", "WindOnshore", "Winddata" + str(year) + ".csv"), index_col=0
        )  # Load Wind Onshore ts
        WindOffdata = pd.read_csv(
            os.path.join(cwd, "InputData", "WindOffshore", "Offshoredata" + str(year) + ".csv"), index_col=0
        )  # Load Wind Offshore ts
        PVdata = pd.read_csv(
            os.path.join(cwd, "InputData", "PV", "PVdata" + str(year) + ".csv"), index_col=0
        )  # Load PV ts

        Operational_data["Wind Onshore " + str(year)] = WindOndata.sum(axis=1) / 38
        Operational_data["Wind Offshore " + str(year)] = WindOffdata.sum(axis=1) / 38
        Operational_data["PV " + str(year)] = PVdata.sum(axis=1) / 38
    
    for cluster in clusters:
        print("Working on clustering into: ", cluster)
        for year in years:
            conn = np.diag(np.ones((1, 8759))[0], -1) + np.diag(np.ones((1, 8759))[0], 1)
            conn[0][-1] = 1
            conn[-1][0] = 1
            _df = pd.DataFrame(
                Operational_data,
                columns=[
                    "Wind Onshore " + str(year),
                    "Wind Offshore " + str(year),
                    "PV " + str(year),
                    "Electricity demand normalized",
                ],
            )
            m = AgglomerativeClustering(
                n_clusters=cluster, distance_threshold=None, connectivity=conn
            )
            out = m.fit_predict(X=_df)
            Operational_data["cluster " + str(year) + " " + str(cluster)] = m.labels_
    
    Yearly_timeperiods_clustered = {}
    Critical_time_periods = pd.DataFrame(columns=["Reference year", "Cluster", "Weather year", "Demand year", "Minimum hourly supply gap", "Duration period"])

    for cluster in clusters:
        Yearly_timeperiods_clustered[cluster] = {}
        for year in years:
            gk = Operational_data.groupby("cluster " + str(year) + " " + str(cluster))
            Yearly_timeperiods_clustered[cluster][year] = gk       

    commodities = [
        "Wind (onshore)",
        "Wind (offshore)",
        "Open field PV",
        "RoofTop PV",
        "CCGT hydrogen gas",
        "Li-ion batteries",
    ]
    typeoftec = {
        "Wind (onshore)": "SourceSinkModel",
        "Wind (offshore)": "SourceSinkModel",
        "Open field PV": "SourceSinkModel",
        "RoofTop PV": "SourceSinkModel",
        "CCGT hydrogen gas": "ConversionModel",
        "Li-ion batteries": "StorageModel",
    }

    for refyear in referenceyears:
        data = readNetCDFToDatasets(
            path.join(
                cwd,
                "Results", "Base optimization", "Basesolution_"
                + str(refyear)
                + ".nc",
            )
        )["Results"]["0"]

        capacities, TAC = {}, {}
        for commodity in commodities:
            capacities[commodity] = data[typeoftec[commodity]][
                commodity
            ].capacityVariablesOptimum.sum()
            TAC[commodity] = zeros_like(capacities[commodity])

        for cluster in Yearly_timeperiods_clustered.keys():
            print(cluster, refyear)
            for year in Yearly_timeperiods_clustered[cluster].keys():
                print(year)
                for gap in Gap:
                    print(gap)
                    for i in range(len(Yearly_timeperiods_clustered[cluster][year])):
                        GenandDemand = (
                            Yearly_timeperiods_clustered[cluster][year].get_group(i).sum()
                        )
                        Existing_caps = [
                            float(
                                capacities["Wind (onshore)"] * GenandDemand["Wind Onshore " + str(year)]
                            ),
                            float(
                                capacities["Wind (offshore)"]
                                * GenandDemand["Wind Offshore " + str(year)]
                            ),
                            float(
                                (capacities["Open field PV"] + capacities["RoofTop PV"])
                                * GenandDemand["PV " + str(year)]
                            ),
                            float(capacities["Li-ion batteries"]),
                            float(capacities["CCGT hydrogen gas"])
                            * len(Yearly_timeperiods_clustered[cluster][year].get_group(i)),
                        ]
                        Supplygap = (
                            GenandDemand["Electricity demand"]
                            - sum(
                                [
                                    capacities["Wind (onshore)"]
                                    * GenandDemand["Wind Onshore " + str(year)],
                                    capacities["Wind (offshore)"]
                                    * GenandDemand["Wind Offshore " + str(year)],
                                    (capacities["Open field PV"] + capacities["RoofTop PV"])
                                    * GenandDemand["PV " + str(year)],
                                    capacities["Li-ion batteries"],
                                ]
                            )
                        ) / len(
                            Yearly_timeperiods_clustered[cluster][year].get_group(i)
                        ) - capacities["CCGT hydrogen gas"]

                        if Supplygap.values > gap:
                            Critical_time_periods.loc[-1] = [refyear, i, year, 2050, gap, len(Yearly_timeperiods_clustered[cluster][year].get_group(i))]
                            Critical_time_periods.index = Critical_time_periods.index + 1

                            print(
                                Supplygap.values,
                                Yearly_timeperiods_clustered[cluster][year]
                                .get_group(i)["cluster " + str(year) + " " + str(cluster)]
                                .iloc[0],
                            )
                            print(
                                "##############"
                                + str(year)
                                + ", "
                                + str(cluster)
                                + "###############"
                            )
                            print(
                                "El demand: ",
                                GenandDemand["Electricity demand"],
                                "\nWind generation: ",
                                float(
                                    capacities["Wind (onshore)"]
                                    * GenandDemand["Wind Onshore " + str(year)]
                                ),
                                float(
                                    capacities["Wind (offshore)"]
                                    * GenandDemand["Wind Offshore " + str(year)]
                                ),
                                "\nPV generation: ",
                                float(
                                    (capacities["Open field PV"] + capacities["RoofTop PV"])
                                    * GenandDemand["PV " + str(year)]
                                ),
                                "\nLi-ion batteries: ",
                                float(capacities["Li-ion batteries"]),
                                "\nDarklullduration: ",
                                len(Yearly_timeperiods_clustered[cluster][year].get_group(i)),
                                "\nCCGT hydrogen gas: ",
                                float(capacities["CCGT hydrogen gas"]),
                                "\nMax total darklull production: ",
                                sum(Existing_caps),
                            )
                            print("##########################")
        
    return Operational_data, Critical_time_periods.sort_index()
    
SimulateYear(1987,fix_assignments=False)
