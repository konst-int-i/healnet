import sys
sys.path.append("/home/kh701/pycharm/x-perceiver/")

import torch
import torch.nn as nn
from torch.autograd.profiler import profile
from sklearn.model_selection import KFold, ParameterGrid
import multiprocessing
import argparse
from argparse import Namespace
import yaml
from tqdm import tqdm
from healnet.train import majority_classifier_acc
from healnet.models.survival_loss import NLLSurvLoss, CrossEntropySurvLoss, CoxPHSurvLoss, nll_loss
from healnet.baselines import RegularizedFCNN
import numpy as np
from torchsummary import summary
import torch_optimizer as t_optim
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch import optim
from healnet.models import HealNet
import pandas as pd
from box import Box
from torch.utils.data import Dataset, DataLoader
from healnet.utils import Config, flatten_config
from healnet.etl import TCGADataset
from pathlib import Path
from datetime import datetime
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)
import wandb



class Pipeline:

    def __init__(self, config: Box, args: Namespace, wandb_name: str=None):
        self.config = flatten_config(config)
        self.args = args
        self.wandb_name = wandb_name
        self.output_dims = int(self.config["model_params.output_dims"])
        self.sources = self.config.sources
        # initialise cuda device (will load directly to GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            print(f"Setting default cuda tensor to double")
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)

        # set up wandb logging
        self.wandb_setup()

    def wandb_setup(self) -> None:

        if args.mode == "sweep":
            with open(self.args.sweep_config, "r") as f:
                sweep_config = yaml.safe_load(f)

            sweep_id = wandb.sweep(sweep=sweep_config, project="x-perceiver")
            wandb.agent(sweep_id, function=self.main)
        else:
            wandb_config = dict(self.config)
            wandb.init(project="x-perceiver", name=self.wandb_name, config=wandb_config)
        return None


    def _check_config(self) -> None:
        """
        Assert that the config only contains valid arguments
        Returns:
            None
        """
        valid_sources = ["omic", "slides"]
        assert all([source in valid_sources for source in self.config.sources]), f"Invalid source specified. Valid sources are {valid_sources}"

        valid_survival_losses = ["nll", "ce_survival", "cox"]
        assert self.config["survival.loss"] in valid_survival_losses, f"Invalid survival loss specified. " \
                                                                   f"Valid losses are {valid_survival_losses}"

        valid_datasets = ["blca", "brca", "kirp", "ucec"]
        assert self.config.dataset in valid_datasets, f"Invalid dataset specified. Valid datasets are {valid_datasets}"

        valid_tasks = ["survival", "classification"]
        assert self.config.task in valid_tasks, f"Invalid task specified. Valid tasks are {valid_tasks}"

        valid_models = ["healnet", "fcnn", "healnet_early"]
        assert self.config.model in valid_models, f"Invalid model specified. Valid models are {valid_models}"

        valid_class_weights = ["inverse", "inverse_root", None]
        assert self.config["model_params.class_weight"] in valid_class_weights, f"Invalid class weight specified. " \
                                                                                f"Valid weights are {valid_class_weights}"

        return None


    def main(self):

        # Initialise wandb run (do here for sweep)
        if self.args.mode == "sweep":
            # update config with sweep config
            wandb.init(project="x-perceiver", name=None) # init sweep run
            for key, value in wandb.config.items():
                if key in self.config.keys():
                    self.config[key] = value

        train_c_indeces, val_c_indeces = [], []
        for fold in range(1, self.config["n_folds"]+1):
            print(f"*****FOLD {fold}*****")
            # fix random seeds for reproducibility
            torch.manual_seed(fold)
            np.random.seed(fold)

            train_data, test_data = self.load_data(fold=fold)
            model = self.make_model(train_data)
            wandb.watch(model)
            if self.config.task == "survival":
                _, train_c_index, _, val_c_index = self.train_survival_fold(model, train_data, test_data, fold)
                train_c_indeces.append(train_c_index)
                val_c_indeces.append(val_c_index)
            elif self.config.task == "classification":
                self.train_clf(model, train_data, test_data)
        # log average and standard deviation across folds
        wandb.log({"mean_train_c_index": np.mean(train_c_indeces),
                   "mean_val_c_index": np.mean(val_c_indeces),
                   "std_train_c_index": np.std(train_c_indeces),
                    "std_val_c_index": np.std(val_c_indeces)})

        wandb.finish()

    def load_data(self, fold: int = None) -> tuple:
        data = TCGADataset(self.config["dataset"],
                           self.config,
                           level=int(self.config["data.wsi_level"]),
                           survival_analysis=True,
                           sources=self.sources,
                           n_bins=self.output_dims,
                           )

        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        print(f"Train samples: {train_size}, Test samples: {test_size}")
        train, test = torch.utils.data.random_split(data, [train_size, test_size])


        # calculate class weights
        if self.config["model_params.class_weights"] == None:
            self.class_weights = None
        else:
            self.class_weights = torch.Tensor(self._calc_class_weights(train)).float().to(self.device)

        train_data = DataLoader(train,
                                batch_size=self.config["train_loop.batch_size"],
                                shuffle=True,
                                num_workers=multiprocessing.cpu_count(),
                                pin_memory=True,
                                multiprocessing_context=MP_CONTEXT,
                                prefetch_factor=2
                                )

        test_data = DataLoader(test,
                               batch_size=self.config["train_loop.batch_size"],
                               shuffle=False,
                               num_workers=multiprocessing.cpu_count(),
                               pin_memory=True,
                               multiprocessing_context=MP_CONTEXT,
                               prefetch_factor=2)
        return train_data, test_data

    def _calc_class_weights(self, train):

        if self.config["model_params.class_weights"] in ["inverse", "inverse_root"]:
            train_targets = np.array(train.dataset.y_disc)[train.indices]
            _, counts = np.unique(train_targets, return_counts=True)
            if self.config["model_params.class_weights"] == "inverse":
                class_weights = 1. / counts
            elif self.config["model_params.class_weights"] == "inverse_root":
                class_weights = 1. / np.sqrt(counts)
        else:
            class_weights = None
        return class_weights

    def make_model(self, train_data: DataLoader):
        """
        Instantiates model and moves to CUDA device if available
        Args:
            train_data:

        Returns:
            nn.Module: model used for training
        """
        feat, _, _, _ = next(iter(train_data))
        if self.config.model == "healnet":

            modalities = len(self.config["sources"])
            if modalities == 1:
                input_channels = [feat[0].shape[2]]
                input_axes = [1]
            elif modalities == 2:
                input_channels = [feat[0].shape[2], feat[1].shape[2]]
                input_axes = [1, 1]
            model = HealNet(
                modalities=modalities,
                input_channels=input_channels, # number of features as input channels
                input_axes=input_axes, # second axis (b n_feats c)
                num_freq_bands=self.config["model_params.num_freq_bands"],
                depth=self.config["model_params.depth"],
                max_freq=self.config["model_params.max_freq"],
                num_classes=self.output_dims, # survival analysis expecting n_bins as output dims
                num_latents = self.config["model_params.num_latents"],
                latent_dim = self.config["model_params.latent_dim"],
                cross_dim_head = self.config["model_params.cross_dim_head"],
                latent_dim_head = self.config["model_params.latent_dim_head"],
                cross_heads = self.config["model_params.cross_heads"],
                latent_heads = self.config["model_params.latent_heads"],
                attn_dropout = self.config["model_params.attn_dropout"],  # non-default
                ff_dropout = self.config["model_params.ff_dropout"],  # non-default
                weight_tie_layers = self.config["model_params.weight_tie_layers"],
                fourier_encode_data = self.config["model_params.fourier_encode_data"],
                self_per_cross_attn = self.config["model_params.self_per_cross_attn"],
                final_classifier_head = True
            )
            model.float()
            model.to(self.device)
            # summary(model, input_size=[feat[0].shape[1:], feat[1].shape[1:]])

        elif self.config.model == "healnet_early":
            modalities = 1 # same model just single modality
            input_channels = [feat[0].shape[2]]
            input_axes = [1]
            model = HealNet(
                modalities=modalities,
                input_channels=input_channels, # number of features as input channels
                input_axes=input_axes, # second axis (b n_feats c)
                num_freq_bands=self.config["model_params.num_freq_bands"],
                depth=self.config["model_params.depth"],
                max_freq=self.config["model_params.max_freq"],
                num_classes=self.output_dims, # survival analysis expecting n_bins as output dims
                num_latents = self.config["model_params.num_latents"],
                latent_dim = self.config["model_params.latent_dim"],
                cross_dim_head = self.config["model_params.cross_dim_head"],
                latent_dim_head = self.config["model_params.latent_dim_head"],
                cross_heads = self.config["model_params.cross_heads"],
                latent_heads = self.config["model_params.latent_heads"],
                attn_dropout = self.config["model_params.attn_dropout"],  # non-default
                ff_dropout = self.config["model_params.ff_dropout"],  # non-default
                weight_tie_layers = self.config["model_params.weight_tie_layers"],
                fourier_encode_data = self.config["model_params.fourier_encode_data"],
                self_per_cross_attn = self.config["model_params.self_per_cross_attn"],
                final_classifier_head = True
            )
            model.float()
            model.to(self.device)

        elif self.config.model == "fcnn":
            model = RegularizedFCNN(output_dim=self.output_dims)
            model.to(self.device)
        return model


    def train_clf(self,
                  model: nn.Module,
                  train_data: DataLoader,
                  test_data: DataLoader,
                  **kwargs):
        """
        Trains model and evaluates classification model
        Args:
            model:
            train_data:
            test_data:
            **kwargs:
        Returns:
        """
        print(f"Training classification model")
        optimizer = optim.SGD(model.parameters(),
                              lr=self.config["optimizer.lr"],
                              momentum=self.config["optimizer.momentum"],
                              weight_decay=self.config["optimizer.weight_decay"]
                              )
        # set efficient OneCycle scheduler, significantly reduces required training iters
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                  max_lr=self.config["optimizer.max_lr"],
                                                  epochs=self.config["train_loop.epochs"],
                                                  steps_per_epoch=len(train_data))


        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # use survival loss for survival analysis which accounts for censored data
        model.train()

        majority_train_acc = np.round(majority_classifier_acc(train_data.dataset.dataset.y_disc), 5)

        for epoch in range(self.config["train_loop.epochs"]):
            print(f"Epoch {epoch}")
            running_loss = 0.0
            predictions = []
            labels = []
            for batch, (features, _, _, y_disc) in enumerate(tqdm(train_data)):
                # only move to GPU now (use CPU for preprocessing)
                labels.append(y_disc.tolist())
                y_disc = y_disc.to(self.device)
                features = features.to(self.device)
                # features, y_disc = features.to(self.device), y_disc.to(self.device)
                if batch == 0 and epoch == 0: # print model summary
                    print(features.shape)
                    print(features.dtype)
                optimizer.zero_grad()
                # forward + backward + optimize

                outputs = model.forward(features)
                loss = criterion(outputs, y_disc)
                # temporary

                loss.backward()
                optimizer.step()
                scheduler.step()
                # print statistics
                running_loss += loss.item()
                predictions.append(outputs.argmax(1).cpu().tolist())

            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            train_loss = np.round(running_loss / len(train_data), 5)
            train_acc = np.round(accuracy_score(y_true=labels, y_pred=predictions), 5)
            train_f1 = np.round(f1_score(y_true=labels, y_pred=predictions, average="weighted"), 5)
            train_confusion_matrix = confusion_matrix(y_true=labels, y_pred=predictions)
            # train_auc = np.round(roc_auc_score(y_true=epoch_labels, y_score=epoch_predictions, average="weighted", multi_class="ovr"), 5)
            # predict entire train set
            print(f"Batch {batch+1}, train_loss: {train_loss}, "
                  f"train_acc: {train_acc}, "
                  f"train_f1: {train_f1}, "
                  f"majority_train_acc: {majority_train_acc}")
            print(f"train_confusion_matrix: \n {train_confusion_matrix}")
            wandb.log({"train_loss": train_loss,
                       "train_acc": train_acc,
                       "train_f1": train_f1,
                       # "majority_train_acc": majority_train_acc
                       }, step=epoch)
            wandb.log({"train_conf_matrix": wandb.plot.confusion_matrix(y_true=labels, preds=predictions)}, step=epoch)
            running_loss = 0.0

            if epoch % self.config["train_loop.eval_interval"] == 0:
                # print("**************************")
                # print(f"EPOCH {epoch} EVALUATION")
                # print("**************************")
                self.evaluate_clf_epoch(model, test_data, criterion, epoch)


    def evaluate_clf_epoch(self, model: nn.Module, test_data: DataLoader, criterion: nn.Module, epoch: int):
        model.eval()
        majority_val_acc = majority_classifier_acc(y_true=test_data.dataset.dataset.y_disc)
        val_loss = 0.0
        val_acc = 0.0
        predictions = []
        labels = []
        with torch.no_grad():
            for batch, (features, _, _, y_disc) in enumerate(test_data):
                labels.append(y_disc.tolist())
                features, y_disc = features.to(self.device), y_disc.to(self.device)
                outputs = model.forward(features)
                loss = criterion(outputs, y_disc)
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == y_disc).sum().item()
                predictions.append(outputs.argmax(1).cpu().tolist())
        val_loss = np.round(val_loss / len(test_data), 5)
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        val_acc = np.round(accuracy_score(labels, predictions), 5)
        val_f1 = np.round(f1_score(labels, predictions, average="weighted"), 5)
        val_conf_matrix = confusion_matrix(labels, predictions)
        print(f"val_loss: {val_loss}, "
              f"val_acc: {val_acc}, "
              f"val_f1: {val_f1}, "
              f"majority_test_acc: {majority_val_acc}")
        print(f"val_conf_matrix: \n {val_conf_matrix}")
        wandb.log({"val_loss": val_loss,
                   "val_acc": val_acc,
                   "val_f1": val_f1,
                   })
        wandb.log({"val_conf_matrix": wandb.plot.confusion_matrix(y_true=labels, preds=predictions)}, step=epoch)
        model.train()


    def train_survival_fold(self,
                            model: nn.Module,
                            train_data: DataLoader,
                            test_data: DataLoader,
                            fold: int = 1,
                            # loss_reg: float = 0.0,
                            gc: int = 16,
                            **kwargs):
        """
        Trains model for survival analysis
        Args:
            model:
            train_data:
            test_data:
            **kwargs:

        Returns:

        """
        print(f"Training survival model using {self.config.model}")
        optimizer = t_optim.lamb.Lamb(model.parameters(), lr=self.config["optimizer.lr"])
        # set efficient OneCycle scheduler, significantly reduces required training iters
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                  max_lr=self.config["optimizer.max_lr"],
                                                  epochs=self.config["train_loop.epochs"],
                                                  steps_per_epoch=len(train_data))

        model.train()

        for epoch in range(1, self.config["train_loop.epochs"]+1):
            print(f"Epoch {epoch}")
            risk_scores = []
            censorships = []
            event_times = []
            # train_loss_surv, train_loss = 0.0, 0.0
            train_loss = 0.0

            for batch, (features, censorship, event_time, y_disc) in enumerate(tqdm(train_data)):
                # only move to GPU now (use CPU for preprocessing)
                features = [feat.to(self.device) for feat in features] # features available for patient
                censorship = censorship.to(self.device) # status 0 or 1
                event_time = event_time.to(self.device) # survival months (continuous)
                y_disc = y_disc.to(self.device) # discretized survival time bucket

                if batch == 0 and epoch == 0: # print model summary
                    print(f"Modality shapes: ")
                    [print(feat.shape) for feat in features]
                    print(f"Modality dtypes:")
                    [print(feat.dtype) for feat in features]

                optimizer.zero_grad()
                # forward + backward + optimize
                logits = model.forward(features)
                y_hat = torch.topk(logits, k=1, dim=1)[1]
                hazards = torch.sigmoid(logits)  # sigmoid to get hazards from predictions for surv analysis
                survival = torch.cumprod(1-hazards, dim=1)  # as per paper, survival = cumprod(1-hazards)
                risk = -torch.sum(survival, dim=1).detach().cpu().numpy()  # risk = -sum(survival)

                if self.config["survival.loss"] == "nll":
                    # loss_fn = NLLSurvLoss()
                    # loss = loss_fn(h=hazards, y=y_disc, c=censorship)
                    loss = nll_loss(hazards=hazards, S=survival, Y=y_disc, c=censorship, weights=self.class_weights)
                elif self.config["survival.loss"] == "ce_survival":
                    loss_fn = CrossEntropySurvLoss()
                    loss = loss_fn(hazards=hazards, survival=survival, y_disc=y_disc, censorship=censorship)
                elif self.config["survival.loss"] == "cox":
                    loss_fn = CoxPHSurvLoss()
                    loss_fn(hazards=hazards, survival=survival, censorship=censorship)

                l1_norm = sum(p.abs().sum() for p in model.parameters())
                reg_loss = float(self.config["optimizer.l1"]) * l1_norm

                # log risk, censorship and event time for concordance index
                risk_scores.append(risk)
                censorships.append(censorship.detach().cpu().numpy())
                event_times.append(event_time.detach().cpu().numpy())

                loss_value = loss.item()
                # train_loss_surv += loss_value
                train_loss += loss_value + reg_loss
                # backward pass
                loss = loss / gc + reg_loss # gradient accumulation step
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            train_loss /= len(train_data)

            risk_scores_full = np.concatenate(risk_scores)
            censorships_full = np.concatenate(censorships)
            event_times_full = np.concatenate(event_times)

            # calculate epoch-level concordance index
            train_c_index = concordance_index_censored((1-censorships_full).astype(bool), event_times_full, risk_scores_full, tied_tol=1e-08)[0]
            wandb.log({f"fold_{fold}_train_loss": train_loss, f"fold_{fold}_train_c_index": train_c_index}, step=epoch)
            print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss, train_c_index))

            # evaluate at interval or if final epoch
            if epoch % self.config["train_loop.eval_interval"] == 0 or epoch == self.config["train_loop.epochs"] - 1:
                val_loss, val_c_index = self.evaluate_survival_epoch(epoch, model, test_data)
                wandb.log({f"fold_{fold}_val_loss": val_loss, f"fold_{fold}_val_c_index": val_c_index}, step=epoch)

        # return values of final epoch
        return train_loss, train_c_index, val_loss, val_c_index

            # checkpoint model after epoch
            # if epoch % self.config["train_loop.checkpoint_interval"] == 0:
            #     torch.save(model.state_dict(), f"{self.log_path}/model_epoch_{epoch}.pt")

    def evaluate_survival_epoch(self,
                                epoch: int,
                                model: nn.Module,
                                test_data: DataLoader,
                                # loss_reg: float=0.0,
                                **kwargs):

        print(f"Running validation...")
        model.eval()
        risk_scores = []
        censorships = []
        event_times = []
        predictions = []
        labels = []
        val_loss_surv, val_loss = 0.0, 0.0

        for batch, (features, censorship, event_time, y_disc) in enumerate(tqdm(test_data)):
            # only move to GPU now (use CPU for preprocessing)
            features = [feat.to(self.device) for feat in features]
            censorship = censorship.to(self.device)
            event_time = event_time.to(self.device)
            y_disc = y_disc.to(self.device)

            y_hat = model.forward(features)
            hazards = torch.sigmoid(y_hat)
            survival = torch.cumprod(1-hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()

            if self.config["survival.loss"] == "nll":
                loss_fn = NLLSurvLoss()
                loss = loss_fn(h=y_hat, y=y_disc, c=censorship)
            elif self.config["survival.loss"] == "ce_survival":
                loss_fn = CrossEntropySurvLoss()
                loss = loss_fn(hazards=hazards, survival=survival, y_disc=y_disc, censorship=censorship)
            elif self.config["survival.loss"] == "cox":
                loss_fn = CoxPHSurvLoss()
                loss_fn(hazards=hazards, survival=survival, censorship=censorship)

            # reg_loss = regularisation_loss(model, l1= self.config["optimizer.l1"], l2=self.config["optimizer.l2"])
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            reg_loss = float(self.config["optimizer.l1"]) * l1_norm

            # log risk, censorship and event time for concordance index
            risk_scores.append(risk)
            censorships.append(censorship.detach().cpu().numpy())
            event_times.append(event_time.detach().cpu().numpy())

            loss_value = loss.item()
            val_loss_surv += loss_value
            val_loss += loss_value + reg_loss

            predictions.append(y_hat.argmax(1).cpu().tolist())
            labels.append(y_disc.detach().cpu().tolist())

        # calculate epoch-level stats
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        val_loss_surv /= len(test_data)
        val_loss /= len(test_data)

        risk_scores_full = np.concatenate(risk_scores)
        censorships_full = np.concatenate(censorships)
        event_times_full = np.concatenate(event_times)

        # calculate epoch-level concordance index
        val_c_index = concordance_index_censored((1-censorships_full).astype(bool), event_times_full, risk_scores_full)[0]
        # f1 = f1_score(labels, predictions, average="macro")
        print('Epoch: {}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss, val_c_index))
        # print(f"Epoch: {epoch}, val_loss: {np.round(val_loss.cpu().detach().numpy(), 5)}, "
        #       f"val_c_index: {np.round(c_index, 5)}")

        model.train()
        return val_loss, val_c_index


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run main training pipeline of x-perceiver")

    # assumes execution
    parser.add_argument("--config_path", type=str, default="/home/kh701/pycharm/x-perceiver/config/main_gpu.yml", help="Path to config file")
    parser.add_argument("--mode", type=str, default="single_run", choices=["single_run", "sweep", "run_plan"])
    parser.add_argument("--sweep_config", type=str, default="config/sweep_bayesian.yaml", help="Hyperparameter sweep configuration")

    # call config
    args = parser.parse_args()
    MP_CONTEXT = "fork"
    # set up multiprocessing context for PyTorch
    torch.multiprocessing.set_start_method(MP_CONTEXT)
    config_path = args.config_path
    config = Config(config_path).read()

    if args.mode == "run_plan":
        # grid = ParameterGrid(
        #     {"dataset": ["blca", "brca", "ucec", "kirp"],
        #      "sources": [["omic"], ["slides"], ["omic", "slides"]],
        #      "model": ["fcnn", "healnet", "healnet_early"],
             # })
        grid = ParameterGrid(
            {"dataset": ["blca"],
             "sources": [["omic"], ["omic", "slides"]],
             "model": ["healnet"],
             })
        folds = 1

        for iteration, params in enumerate(grid):
            dataset, sources, model = params["dataset"], params["sources"], params["model"]
            print(f"Run plan iteration {iteration+1}/{len(grid)}")
            print(f"Dataset: {dataset}, Sources: {sources}, Model: {model}")
            config["dataset"] = dataset
            config["sources"] = sources
            config["model"] = model
            config["n_folds"] = folds
            pipeline = Pipeline(
                    config=config,
                    args=args,
                )
            pipeline.main()

        print(f"Successfully finished runplan"
              f"{print(list(grid))}")

    else: # single_run or sweep
        pipeline = Pipeline(
                    config=config,
                    args=args,
                )
        pipeline.main()