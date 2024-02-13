DATASET_PATH = "../data/Boiling ANN data.xlsx"
DUZ_H = "Düz boru H "
DUZ_DP = "Düz Boru dP"
MIKRO_H = "Mikrokanatlı Boru H"
MIKRO_DP = "Mikrokanatlı Boru dP"
DATA_H_INDEX_MAP = {
            0: 'Mass flux',
            1: 'Saturation pressure',
            2: 'Heat flux',
            3: 'Quality',
            4: 'Pressure drop',
            # 5: 'Heat transfer coefficient',
            5: 'Reynolds number',
            6: 'Two-phase multiplier',
            7: 'Froude number',
            8: 'Weber number',
            9: 'Bond number',
            10: "Tube type"
        }

DATA_DP_INDEX_MAP = {
            0: 'Mass flux',
            1: 'Saturation pressure',
            2: 'Heat flux',
            3: 'Quality',
            # 4: 'Pressure drop',
            4: 'Heat transfer coefficient',
            5: 'Reynolds number',
            6: 'Two-phase multiplier',
            7: 'Froude number',
            8: 'Weber number',
            9: 'Bond number',
            10: "Tube type"
        }
OUTPUT_MULTIINDEX_NAME = ("Output", "Output 1")
INPUT_LAYER_NAME = 'Input'
INPUT_RAW_LAYER_NAME = 'Input Raw'
INPUT_CALCULATED_LAYER_NAME = 'Input Calculated'

TARGET_H = "Heat transfer coefficient"
TARGET_DP = "Pressure drop"

DATASET_KEY_PLAIN_H = 'Plain tube h'
DATASET_KEY_PLAIN_DP = 'Plain tube dp'
DATASET_KEY_MICRO_H = 'Microfin tube h'
DATASET_KEY_MICRO_DP = 'Microfin tube dp'
