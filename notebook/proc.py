from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

from utils import Dataset, getDataset
import utils as u
import conf as cfg
import pandas as pd
import seaborn as sns
import time
import sklearnex
import logging
idx = pd.IndexSlice
pd.set_option('display.max_columns', 800)
pd.set_option('display.max_rows', 800)

sklearnex.patch_sklearn()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


ENABLE_HTP = True
ENABLE_DP = True
ENABLE_EXTRA_HTP = True
ENABLE_EXTRA_DP = True


if __name__ == "__main__":

    # %% Data Processing
    then = time.time()
    logger = logging.getLogger(__name__)
    logger.info("Process started.")
    logger.info("Reading excel file...")

    datasets_processed = {}
    datasets_processed[cfg.DATASET_KEY_PLAIN_H] = getDataset(pd.read_excel(cfg.DATASET_PATH, sheet_name=cfg.DUZ_H, header=None), raw_layer_name=cfg.INPUT_RAW_LAYER_NAME, calculated_layer_name=cfg.INPUT_CALCULATED_LAYER_NAME)
    datasets_processed[cfg.DATASET_KEY_MICRO_H] = getDataset(pd.read_excel(cfg.DATASET_PATH, sheet_name=cfg.MIKRO_H, header=None), raw_layer_name=cfg.INPUT_RAW_LAYER_NAME, calculated_layer_name=cfg.INPUT_CALCULATED_LAYER_NAME)
    datasets_processed[cfg.DATASET_KEY_PLAIN_DP] = getDataset(pd.read_excel(cfg.DATASET_PATH, sheet_name=cfg.DUZ_DP, header=None), raw_layer_name=cfg.INPUT_RAW_LAYER_NAME, calculated_layer_name=cfg.INPUT_CALCULATED_LAYER_NAME)
    datasets_processed[cfg.DATASET_KEY_MICRO_DP] = getDataset(pd.read_excel(cfg.DATASET_PATH, sheet_name=cfg.MIKRO_DP, header=None), raw_layer_name=cfg.INPUT_RAW_LAYER_NAME, calculated_layer_name=cfg.INPUT_CALCULATED_LAYER_NAME)
    logger.info("Consolidating data...")

    data_h = pd.concat([
        datasets_processed[cfg.DATASET_KEY_PLAIN_H].assign(dataset=cfg.DATASET_KEY_PLAIN_H),
        datasets_processed[cfg.DATASET_KEY_MICRO_H].assign(dataset=cfg.DATASET_KEY_MICRO_H)
    ], axis=0, ignore_index=True)


    data_dp = pd.concat([
        datasets_processed[cfg.DATASET_KEY_PLAIN_DP].assign(dataset=cfg.DATASET_KEY_PLAIN_DP),
        datasets_processed[cfg.DATASET_KEY_MICRO_DP].assign(dataset=cfg.DATASET_KEY_MICRO_DP)
    ], axis=0, ignore_index=True)

    logger.info("h_TP data shape: {}".format(data_h.shape))
    logger.info("dp_TP data shape: {}".format(data_dp.shape))

    Xh = data_h.loc[:, idx[[cfg.INPUT_RAW_LAYER_NAME, cfg.INPUT_CALCULATED_LAYER_NAME], :]]
    yh = data_h[cfg.OUTPUT_MULTIINDEX_NAME[0]]

    Xdp = data_dp.loc[:, idx[[cfg.INPUT_RAW_LAYER_NAME, cfg.INPUT_CALCULATED_LAYER_NAME], :]]
    ydp = data_dp[cfg.OUTPUT_MULTIINDEX_NAME[0]]

    # %% Create samples
    h_sample = data_h.T.drop(cfg.OUTPUT_MULTIINDEX_NAME).reset_index(drop=True).rename(
        index=cfg.DATA_H_INDEX_MAP,
    ).T.assign(y=yh).rename(columns={'y': cfg.TARGET_H}).sample(5).head()
    logger.info("Microfinned and Plain Tubes - h")
    print(h_sample)


    # Rename using definitions in the markdown

    logger.info("Microfinned and Plain Tubes - $\Delta P$")
    dp_sample = data_dp.T.drop(cfg.OUTPUT_MULTIINDEX_NAME).reset_index(drop=True).rename(
        index=cfg.DATA_DP_INDEX_MAP,
    ).T.assign(y=ydp).rename(columns={'y': cfg.TARGET_DP}).sample(5).head()
    print(dp_sample)

    #%% Define CV and models config
    cv = u.ArbitraryStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_fn_mapping = [
        u.ModelConfig(model_name='ANN Raw', model_feature_subset=u.getRaw, model_getter=u.getANNRandomSearch, cv_config=None),
        u.ModelConfig(model_name='ANN Calc', model_feature_subset=u.getProcessed, model_getter=u.getANNRandomSearch, cv_config=None),
        u.ModelConfig(model_name='ANN Raw sCV', model_feature_subset=u.getRaw, model_getter=u.getANNRandomSearch, cv_config=cv),
        u.ModelConfig(model_name='ANN Calc sCV', model_feature_subset=u.getProcessed, model_getter=u.getANNRandomSearch, cv_config=cv),
        u.ModelConfig(model_name='LWR Raw', model_feature_subset=u.getRaw, model_getter=u.getLwrExtendedNeighbors, cv_config=None),
        u.ModelConfig(model_name='LWR Calc', model_feature_subset=u.getProcessed, model_getter=u.getLwrExtendedNeighbors, cv_config=None),
        u.ModelConfig(model_name='LWR Raw sCV', model_feature_subset=u.getRaw, model_getter=u.getLwrExtendedNeighbors, cv_config=cv),
        u.ModelConfig(model_name='LWR Calc sCV', model_feature_subset=u.getProcessed, model_getter=u.getLwrExtendedNeighbors, cv_config=cv),
        u.ModelConfig(model_name='GBM Raw', model_feature_subset=u.getRaw, model_getter=u.getGBMFocused),
        u.ModelConfig(model_name='GBM Calc', model_feature_subset=u.getProcessed, model_getter=u.getGBMFocused),
        u.ModelConfig(model_name='GBM Raw sCV', model_feature_subset=u.getRaw, model_getter=u.getGBMFocused, cv_config=cv),
        u.ModelConfig(model_name='GBM Calc sCV', model_feature_subset=u.getProcessed, model_getter=u.getGBMFocused, cv_config=cv),
    ]

    #%% Conduct HTP
    if ENABLE_HTP:
        logger.info("Conducting hTP experiments...")
        htp = u.Dataset(data=data_h, X=Xh, y=yh)
        htp.setRenameFunc(u.renameH)
        htp.setColNames(u.H_NAMES)
        htp.setDec(u.getDecPipeline())
        logger.info("Performing decomposition...")
        htp.decompose()
        htp.setClusterer(u.getClusterer())
        logger.info("Performing clustering...")
        htp.setCategories()
        logger.info("Performing train-test split...")
        htp.split(random_state=42, test_size=0.2)
        htp.defineCats()
        htp.setScoring(u.SCORING)



        logger.info("Training models...")
        for mc_obj in model_fn_mapping:
            logger.info(mc_obj.model_name)
            htp.addModel(mc_obj.model_getter(cv=mc_obj.cv_config, subselector_fn=mc_obj.model_feature_subset), mc_obj.model_name)

        logger.info("Processing results...")
        for m in htp.models_cv:
            print(m)
            _ = u.process_ann_results(htp.models_cv[m], m)[0]
            _.to_csv("../output/htp_normal_{}.csv".format(m), index=True)
        htp.to_pickle("../output/htp.pkl")
        logger.info("htp saved")

    #%% Conduct DP
    if ENABLE_DP:
        logger.info("Conducting dpTP experiments...")
        dp = u.Dataset(data=data_dp, X=Xdp, y=ydp)
        dp.setRenameFunc(u.renameDP)
        dp.setColNames(u.DP_NAMES)
        dp.setDec(u.getDecPipeline())
        logger.info("Performing decomposition...")
        dp.decompose()
        dp.setClusterer(u.getClusterer())
        logger.info("Performing clustering...")
        dp.setCategories()
        logger.info("Performing train-test split...")
        dp.split(random_state=42, test_size=0.2)
        dp.defineCats()
        dp.setScoring(u.SCORING)

        logger.info("Training models...")

        for mc_obj in model_fn_mapping:
            logger.info(mc_obj.model_name)
            dp.addModel(mc_obj.model_getter(cv=mc_obj.cv_config, subselector_fn=mc_obj.model_feature_subset), mc_obj.model_name)

        logger.info("Processing results...")
        for m in dp.models_cv:
            _ = u.process_ann_results(dp.models_cv[m], m)[0]
            _.to_csv("../output/dp_normal_{}.csv".format(m), index=True)
        dp.to_pickle("../output/dp.pkl")
        logger.info("dp saved")

    #%% Conduct HTP outliers
    if ENABLE_EXTRA_HTP:
        logger.info("Conducting hTP experiments with outliers...")
        extra_htp = u.OutlierDataset(data=data_h, X=Xh, y=yh)
        extra_htp.setRenameFunc(u.renameH)
        extra_htp.setColNames(u.H_NAMES)
        extra_htp.setDec(u.getDecPipeline())
        extra_htp.decompose()
        extra_htp.setClusterer(u.getClusterer())
        extra_htp.setCategories()
        extra_htp.split()
        extra_htp.setScoring(u.SCORING)

        logger.info("Training models...")
        for mc_obj in model_fn_mapping:
            if mc_obj.cv_config is not None:
                continue
            logger.info(mc_obj.model_name)
            extra_htp.addModel(mc_obj.model_getter(subselector_fn=mc_obj.model_feature_subset), mc_obj.model_name)

        logger.info("Processing results...")
        for m in extra_htp.models_cv:
            _ = u.process_ann_results(extra_htp.models_cv[m], m)[0]
            _.to_csv("../output/htp_outliers_{}.csv".format(m), index=True)
        extra_htp.to_pickle("../output/extra_htp.pkl")
        logger.info("extra_htp saved")

    # %% Conduct DP outliers
    if ENABLE_EXTRA_DP:
        logger.info("Conducting dp experiments with outliers...")
        extra_dp = u.OutlierDataset(data=data_dp, X=Xdp, y=ydp)
        extra_dp.setRenameFunc(u.renameDP)
        extra_dp.setColNames(u.DP_NAMES)
        extra_dp.setDec(u.getDecPipeline())
        logger.info("Performing decomposition...")
        extra_dp.decompose()
        extra_dp.setClusterer(u.getClusterer())
        logger.info("Performing clustering...")
        extra_dp.setCategories()
        logger.info("Performing train-test split...")
        extra_dp.split()
        extra_dp.setScoring(u.SCORING)


        logger.info("Training models...")
        for mc_obj in model_fn_mapping:
            if mc_obj.cv_config is not None:
                continue
            logger.info(mc_obj.model_name)
            extra_dp.addModel(mc_obj.model_getter(subselector_fn=mc_obj.model_feature_subset), mc_obj.model_name)

        logger.info("Processing results...")
        for m in extra_dp.models_cv:
            _ = u.process_ann_results(extra_dp.models_cv[m], m)[0]
            _.to_csv("../output/dp_outliers_{}.csv".format(m), index=True)
        extra_dp.to_pickle("../output/extra_dp.pkl")
        logger.info("extra_dp saved")

    # %% Save artifacts
    logger.info("Saving data...")
    try:
        pass
    except Exception as e:
        logger.error("Error saving data")
        # Print exception with traceback
        print(e)
        import traceback
        traceback.print_exc()

    now = time.time()
    logger.info("Process completed.")
    print("Complete")
    print(f"Time elapsed: {now-then:.2f} seconds")


