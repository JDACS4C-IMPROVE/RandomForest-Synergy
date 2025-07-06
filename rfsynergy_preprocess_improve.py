import sys
from pathlib import Path
import pandas as pd

# Core improvelib imports
import improvelib.utils as frm
# Application-specific (DRP) imports
from improvelib.applications.synergy.config import SynergyPreprocessConfig

# Model-specifc imports
from model_params_def import preprocess_params

filepath = Path(__file__).resolve().parent


# [Req]
def run(params):
    # ------------------------------------------------------
    # [Req] Load feature data
    # ------------------------------------------------------
    print("Load omics data.")
    omics = frm.get_x_data(file = params['cell_transcriptomic_file'], 
                                        benchmark_dir = params['input_dir'], 
                                        column_name = params['canc_col_name'])

    print("Load drug data.")
    drugs = frm.get_x_data(file = params['drug_mordred_file'], 
                    benchmark_dir = params['input_dir'], 
                    column_name = params['drug_col_name'])
    
    # ------------------------------------------------------
    # [Req] Validity check of feature representations
    # ------------------------------------------------------
    # not needed for this data/model

    # ------------------------------------------------------
    # [Req] Determine preprocessing on training data
    # ------------------------------------------------------
    print("Load train response data.")
    response_train = frm.get_y_data(split_file=params["train_split_file"], 
                                   benchmark_dir=params['input_dir'], 
                                   y_data_file=params['y_data_file'])
    response_train = response_train.dropna(subset=[params['y_col_name']])
    
    print("Find intersection of training data.")
    response_train = frm.get_y_data_with_features(response_train, omics, params['canc_col_name'])
    response_train = frm.get_y_data_with_features(response_train, drugs, params['drug_1_col_name'])
    response_train = frm.get_y_data_with_features(response_train, drugs, params['drug_2_col_name'])
    omics_train = frm.get_features_in_y_data(omics, response_train, params['canc_col_name'])
    drug1_train = frm.get_features_in_y_data(drugs, response_train, params['drug_1_col_name'])
    drug2_train = frm.get_features_in_y_data(drugs, response_train, params['drug_2_col_name'])
    drugs_train = pd.concat([drug1_train, drug2_train]).drop_duplicates()

    print("Determine transformations.")
    frm.determine_transform(omics_train, 'omics_transform', params['cell_transcriptomic_transform'], params['output_dir'])
    frm.determine_transform(drugs_train, 'drugs_transform', params['drug_mordred_transform'], params['output_dir'])


    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------
    # Dict with split files corresponding to the three sets (train, val, and test)
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    for stage, split_file in stages.items():
        print(f"Prepare data for stage {stage}.")
        print(f"Find intersection of {stage} data.")
        response_stage = frm.get_y_data(split_file=split_file, 
                                benchmark_dir=params['input_dir'], 
                                y_data_file=params['y_data_file'])
        response_stage = response_stage.dropna(subset=[params['y_col_name']])
        response_stage = frm.get_y_data_with_features(response_stage, omics, params['canc_col_name'])
        response_stage = frm.get_y_data_with_features(response_stage, drugs, params['drug_1_col_name'])
        response_stage = frm.get_y_data_with_features(response_stage, drugs, params['drug_2_col_name'])
        print("response_stage:", response_stage)
        omics_stage = frm.get_features_in_y_data(omics, response_stage, params['canc_col_name'])
        drug1_stage = frm.get_features_in_y_data(drugs, response_stage, params['drug_1_col_name'])
        drug2_stage = frm.get_features_in_y_data(drugs, response_stage, params['drug_2_col_name'])
        drugs_stage = pd.concat([drug1_stage, drug2_stage]).drop_duplicates()

        print(f"Transform {stage} data.")
        omics_stage = frm.transform_data(omics_stage, 'omics_transform', params['output_dir'])
        drugs_stage = frm.transform_data(drugs_stage, 'drugs_transform', params['output_dir'])

        print(f"Merge {stage} data")
        y_df_cols = response_stage.columns.tolist()
        data = response_stage.merge(omics_stage, on=params["canc_col_name"], how="inner")
        data = data.merge(drugs_stage, left_on=params["drug_1_col_name"], right_on=params['drug_col_name'], how="inner")
        data = data.merge(drugs_stage, left_on=params["drug_2_col_name"], right_on=params['drug_col_name'], how="inner")
        print("data:", data)
        print("nas:", data[params['y_col_name']].isna().sum())
        data = data.sample(frac=1.0).reset_index(drop=True) # shuffle

        print(f"Save {stage} data")
        xdf = data.drop(columns=y_df_cols)
        xdf[params['y_col_name']] = data[params['y_col_name']]
        data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage=stage)
        xdf.to_parquet(Path(params["output_dir"]) / data_fname)
        # [Req] Save y dataframe for the current stage
        ydf = data[y_df_cols]
        frm.save_stage_ydf(ydf, stage, params["output_dir"])

    return params["output_dir"]


# [Req]
def main(args):
    cfg = SynergyPreprocessConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="rfsynergy_params.ini",
                                       additional_definitions=preprocess_params)
    timer_preprocess = frm.Timer()
    ml_data_outdir = run(params)
    timer_preprocess.save_timer(dir_to_save=params["output_dir"], 
                                filename='runtime_preprocess.json', 
                                extra_dict={"stage": "preprocess"})
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])