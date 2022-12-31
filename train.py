from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import deepchem as dc

# TODO Refactor to have a universal interface for arbritrary model and task combinations, if required

def train_rf_model(dataset: dict, mode: str):
    """Trains a random forest model on deepchem datasets

    Args:
        dataset (dict): contains 'train' dataset in deepchem format
        mode (str): classification or regression

    Returns:
        _type_: trained deepchem model
    """
    if mode == "regression":
        rf = RandomForestRegressor()
    elif mode == "classification":
        rf = RandomForestClassifier()
    else:
        raise ValueError("mode must be from {regression, classification}")

    model = dc.models.SklearnModel(model=rf)
    model.fit(dataset["train"])
    return model


def train_graph_model(
    dataset: dict,
    mode: str,
    model_type: str,
    patience: int = 10,
    max_epochs: int = 300,
):
    """_Trains GAT or GCN models on deepchem datasets

    Args:
        dataset (dict): contains 'train' and 'valid' datasets in deepchem format
        mode (str): classification or regression
        model_type (str): 'gat' or 'gcn'
        patience (int, optional): number of epochs to continue training
            after validation loss reaches a minimum. Defaults to 10.
        max_epochs (int, optional): maximum number of epochs in train loop. Defaults to 300.

    Returns:
        _type_: _description_
    """
    if model_type == "gcn":
        model = dc.models.GCNModel(
            mode=mode, n_tasks=1, batch_size=32, learning_rate=0.001
        )
    elif model_type == "gat":
        model = dc.models.GATModel(
            mode=mode, n_tasks=1, batch_size=32, learning_rate=0.001
        )
    else:
        raise ValueError("model_type must be from {gcn, gat}")

    val_losses = []
    losses = []

    n_rounds_worse_loss = 0
    for epoch in range(max_epochs):
        loss = model.fit(
            dataset["train"],
            nb_epoch=1,
            max_checkpoints_to_keep=1,
            all_losses=losses,
            deterministic=True,
        )

        if mode == "regression":
            metric = dc.metrics.Metric(dc.metrics.score_function.rms_score)
            val_loss = (
                model.evaluate(dataset["valid"], metrics=metric)["rms_score"] ** 2
            )
        else:
            metric = dc.metrics.Metric(dc.metrics.score_function.prc_auc_score)
            val_loss = (
                1 - model.evaluate(dataset["valid"], metrics=metric)["prc_auc_score"]
            )  # monitor 1 - AUC-PRC for early stopping in classification tasks

        val_losses.append(val_loss)

        if val_loss > min(val_losses[:-1], default=float("inf")):
            n_rounds_worse_loss += 1
        else:
            n_rounds_worse_loss = 0
        if n_rounds_worse_loss > patience:
            break

    return model
