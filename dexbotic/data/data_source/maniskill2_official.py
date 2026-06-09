import math

from dexbotic.data.data_source.register import register_dataset


MANISKILL2_DATASET = {
    "pickcube": {
        "data_path_prefix": "/dexmal-fa-yjh-data/dex_data/maniskill2/video/",
        "annotations": "/dexmal-fa-yjh-data/dex_data/maniskill2/jsonl/PickCube-v0",
        "frequency": 1,
    },
    "stackcube": {
        "data_path_prefix": "/dexmal-fa-yjh-data/dex_data/maniskill2/video/",
        "annotations": "/dexmal-fa-yjh-data/dex_data/maniskill2/jsonl/StackCube-v0",
        "frequency": 1,
    },
    "picksingleycb": {
        "data_path_prefix": "/dexmal-fa-yjh-data/dex_data/maniskill2/video/",
        "annotations": "/dexmal-fa-yjh-data/dex_data/maniskill2/jsonl/PickSingleYCB-v0",
        "frequency": 1,
    },
    "picksingleegad": {
        "data_path_prefix": "/dexmal-fa-yjh-data/dex_data/maniskill2/video/",
        "annotations": "/dexmal-fa-yjh-data/dex_data/maniskill2/jsonl/PickSingleEGAD-v0",
        "frequency": 1,
    },
    "pickclutterycb": {
        "data_path_prefix": "/dexmal-fa-yjh-data/dex_data/maniskill2/video/",
        "annotations": "/dexmal-fa-yjh-data/dex_data/maniskill2/jsonl/PickClutterYCB-v0",
        "frequency": 1,
    },
    "pi0_all": {
        "data_path_prefix": "/dexmal-fa-yjh-data/dex_data/maniskill2_dex/dexdata/video",
        "annotations": "/dexmal-fa-yjh-data/dex_data/maniskill2_dex/dexdata",
        "frequency": 1,
    },
}

meta_data = {
    'non_delta_mask': [6],
    'periodic_mask': [3, 4, 5],
    'periodic_range': 2 * math.pi,
}

register_dataset(MANISKILL2_DATASET, meta_data=meta_data, prefix='maniskill')