import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# [Req] IMPROVE imports
from improvelib.applications.synergy.config import SynergyTrainConfig
import improvelib.utils as frm
from improvelib.metrics import compute_metrics

# Model-specifc imports
from model_params_def import train_params # [Req]

filepath = Path(__file__).resolve().parent # [Req]


# [Req]
def run(params: Dict) -> Dict:
    # ------------------------------------------------------
    # [Req] Build model path
    # ------------------------------------------------------
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"]
    )

    # ------------------------------------------------------
    # [Req] Build model path and data names for train and val sets
    # ------------------------------------------------------
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"]
    )
    train_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")
    val_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    train_data = pd.read_parquet(Path(params["input_dir"]) / train_data_fname)
    val_data = pd.read_parquet(Path(params["input_dir"]) / val_data_fname)

    # Train data
    ytr = train_data[[params["y_col_name"]]]
    ytr = ytr.values.squeeze()
    xtr = train_data.drop(columns=[params['y_col_name']])

    # Val data
    yvl = val_data[[params["y_col_name"]]]
    xvl = val_data.drop(columns=[params['y_col_name']])
    val_true = yvl.values.squeeze()
    # ------------------------------------------------------
    # Prepare, train, and save model
    # ------------------------------------------------------

    # n_estimators is treated as epochs
    # initial round
    model = RandomForestRegressor(max_depth=None, n_estimators=1)
    model.fit(xtr, ytr)
    val_pred = model.predict(xvl)
    val_metrics = compute_metrics(y_true=val_true, y_pred=val_pred, metric_type=params['metric_type'])
    best_loss = val_metrics[params['loss']]
    joblib.dump(model, str(modelpath))
    rounds = 1
    early_stop = 0
    while rounds < params['epochs'] and early_stop < params['patience']:
        rounds = rounds + 1
        model.set_params(n_estimators=rounds, warm_start=True)
        model.fit(xtr, ytr)
        val_pred = model.predict(xvl)
        val_metrics = compute_metrics(y_true=val_true, y_pred=val_pred, metric_type=params['metric_type'])
        val_loss = val_metrics[params['loss']]
        if val_loss < best_loss:
            early_stop = 0
            best_loss = val_loss
            joblib.dump(model, str(modelpath))
        else:
            early_stop = early_stop + 1
        print(f"Epoch: {rounds}, val_loss: {val_loss}, epochs since improvement: {early_stop}")

    del model

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Load the best saved model (as determined based on val data)
    model = joblib.load(str(modelpath))

    # Compute predictions
    val_pred = model.predict(xvl)
    
   
     # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=val_true, 
        y_pred=val_pred, 
        stage="val",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performance_scores(
        y_true=val_true, 
        y_pred=val_pred, 
        stage="val",
        metric_type=params["metric_type"],
        output_dir=params["output_dir"]
    )

    return val_scores


# [Req]
def main(args):
    cfg = SynergyTrainConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="rfsynergy_params.ini",
                                       additional_definitions=train_params)
    timer_train = frm.Timer()    
    val_scores = run(params)
    timer_train.save_timer(dir_to_save=params["output_dir"], 
                           filename='runtime_train.json', 
                           extra_dict={"stage": "train"})
    print("\nFinished model training.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])