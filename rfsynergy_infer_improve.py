import sys
from pathlib import Path
import pandas as pd

# Core improvelib imports
import improvelib.utils as frm
# Application-specific (DRP) imports
from improvelib.applications.synergy.config import SynergyInferConfig

# Model-specifc imports
from model_params_def import infer_params
from sklearn.ensemble import RandomForestRegressor
import joblib

filepath = Path(__file__).resolve().parent

# [Req]
def run(params):
    # ------------------------------------------------------
    # [Req] Build model path and create data name for test set
    # ------------------------------------------------------
    modelpath = frm.build_model_path(model_file_name=params["model_file_name"],
                                     model_file_format=params["model_file_format"],
                                     model_dir=params["input_model_dir"])
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    test_data = pd.read_parquet(Path(params["input_data_dir"]) / test_data_fname)

    # Test data
    yte = test_data[[params["y_col_name"]]]
    xte = test_data.drop(columns=[params['y_col_name']])

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Load model
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["input_model_dir"]
    ) # [Req]

    model = joblib.load(str(modelpath))

    # Compute predictions
    test_pred = model.predict(xte)
    test_true = yte.values.squeeze()

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=test_true, 
        y_pred=test_pred, 
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_data_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=test_true, 
            y_pred=test_pred, 
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )

    return True


# [Req]
def main(args):
    cfg = SynergyInferConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="rfsynergy_params.ini",
                                       additional_definitions=infer_params)
    timer_infer = frm.Timer()    
    status = run(params)
    timer_infer.save_timer(dir_to_save=params["output_dir"], 
                           filename='runtime_infer.json', 
                           extra_dict={"stage": "infer"})
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])