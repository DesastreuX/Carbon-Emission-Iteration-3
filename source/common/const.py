class FILEPATH:
    DATASET_DIRECTORY_PATH = "../datasets/individual_carbon_footprint_calculation_13"
    RAW = f"{DATASET_DIRECTORY_PATH}/Carbon Emission.csv"
    CARBON_EMISSION_AMOUNT = (
        f"{DATASET_DIRECTORY_PATH}/Carbon Emission Files/Carbon Emission Amount.csv"
    )
    CARBON_EMISSION_HEALTH = (
        f"{DATASET_DIRECTORY_PATH}/Carbon Emission Files/Carbon Emission Health.csv"
    )
    CARBON_EMISSION_LIFESTYLE = (
        f"{DATASET_DIRECTORY_PATH}/Carbon Emission Files/Carbon Emission Lifestyle.csv"
    )
    CARBON_EMISSION_TRAVEL = (
        f"{DATASET_DIRECTORY_PATH}/Carbon Emission Files/Carbon Emission Travel.csv"
    )
    CARBON_EMISSION_WASTE = (
        f"{DATASET_DIRECTORY_PATH}/Carbon Emission Files/Carbon Emission Waste.csv"
    )
    TEMP_FILE_DIRECTORY_PATH = "./temp"
    TEMP_STAGING_PATH = f"{TEMP_FILE_DIRECTORY_PATH}/staging"


class STAGING_FILENAME:
    DP = "03-staging"
    DT = "04-staging"
    DMA = "06-staging"


class DATASET:
    TARGET = "CarbonEmission"
