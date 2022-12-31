import deepchem as dc
import sklearn
import tdc


def make_dc_split(tdc_dataset: tdc.single_pred.adme.ADME, featurizer_type: str) -> dict:
    """
    Converts a TDC dataset into a format suitable for training deepchem models

    Args:
        tdc_dataset (tdc.single_pred.adme.ADME): dataset from TDC
        featurizer_type (str): molecular featurizer type from 'molgraphconv' or 'ecfp'

    Returns:
        dict: contains 'train', 'valid', and 'test' datasets in deepchem dataloader format
    """
    if featurizer_type == "molgraphconv":
        featurizer = dc.feat.MolGraphConvFeaturizer()
    elif featurizer_type == "ecfp":
        featurizer = dc.feat.CircularFingerprint(radius=4, size=1024)
    else:
        raise ValueError("featurizer_type must be from {molgraphconv, ecfp}")
    loader = dc.data.InMemoryLoader(tasks=["task"], featurizer=featurizer)
    dc_dataset = {
        name: loader.create_dataset(zip(data["Drug"], data["Y"]))
        for name, data in tdc_dataset.get_split(method="scaffold").items()
    }
    return dc_dataset


def evaluate(model, dataset: dict, mode: str) -> dict:
    """
    Evaluates a regresison or classification model using scaled RMSE or area under precison recall curve

    Args:
        model: trained model with a .predict() method, such as a trained deepchem model
        dataset (dict): contains 'train', 'valid', 'test' deepchem datasets
        mode (str): 'regression' or 'classification'

    Returns:
        dict: contains score metrics, including 'auc', 'srmse' (standard scaled mean squared error),
    """
    y_pred = model.predict(dataset["test"])
    scores = {}
    if mode == "regression":
        rmse = sklearn.metrics.mean_squared_error(
            dataset["test"].y, y_pred, squared=False
        )
        sd = dataset["test"].y.std()
        scores["srmse"] = rmse / sd
    elif mode == "classification":
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
            dataset["test"].y, y_pred[:, 1]
        )
        auc_value = sklearn.metrics.auc(recall, precision)
        scores["auc"] = auc_value
        scores["prc"] = (precision, recall, thresholds)
    return scores
