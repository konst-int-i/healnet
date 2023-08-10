import sys
sys.path.append("/home/kh701/pycharm/x-perceiver/")

import torch
import torch.nn as nn
import traceback
from torch.autograd.profiler import profile
from sklearn.model_selection import KFold, ParameterGrid
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from functools import partial
import argparse
from itertools import repeat
from argparse import Namespace
import yaml
from tqdm import tqdm
from healnet.train import majority_classifier_acc
from healnet.utils import EarlyStopping, calc_reg_loss, pickle_obj, unpickle
from healnet.models.survival_loss import NLLSurvLoss, CrossEntropySurvLoss, CoxPHSurvLoss, nll_loss
from healnet.models.explainer import Explainer
from healnet.baselines import RegularizedFCNN, MMPrognosis, MCAT, SNN, MILAttentionNet
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
        self.dataset = self.config.dataset
        self.args = args
        self._check_config()
        self.wandb_name = wandb_name
        self.output_dims = int(self.config[f"model_params.output_dims"])
        self.sources = self.config.sources
        # create log directory for run
        # date
        self.local_run_id = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        # initialise cuda device (will load directly to GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            print(f"Setting default cuda tensor to double")
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)

        # set up wandb logging
        self.wandb_setup()

        if self.config.explainer:
            self.log_dir = Path(self.config.log_path).joinpath(f"{wandb.run.name}")
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def wandb_setup(self) -> None:

        if self.args.mode == "sweep":
            with open(self.args.sweep_config, "r") as f:
                sweep_config = yaml.safe_load(f)

            sweep_id = wandb.sweep(sweep=sweep_config, project="x-perceiver")
            wandb.agent(sweep_id, function=self.main)
        else:
            wandb_config = dict(self.config)
            wandb.init(project="x-perceiver", name=self.wandb_name, config=wandb_config, resume=True)
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

        valid_datasets = ["blca", "brca", "kirp", "ucec", "hnsc", "paad", "luad", "lusc"]
        assert self.config.dataset in valid_datasets, f"Invalid dataset specified. Valid datasets are {valid_datasets}"

        # # check that model parameters are specified
        # assert self.config.dataset in self.config.model_params.keys(), f"Model parameters not specified for dataset {self.config.dataset}"

        valid_tasks = ["survival", "classification"]
        assert self.config.task in valid_tasks, f"Invalid task specified. Valid tasks are {valid_tasks}"

        valid_models = ["healnet", "fcnn", "healnet_early", "mcat", "mm_prognosis"]
        assert self.config.model in valid_models, f"Invalid model specified. Valid models are {valid_models}"

        valid_class_weights = ["inverse", "inverse_root", None]
        assert self.config[f"model_params.class_weights"] in valid_class_weights, f"Invalid class weight specified. " \
                                                                                f"Valid weights are {valid_class_weights}"

        return None


    def main(self):

        # Initialise wandb run (do here for sweep)
        if self.args.mode == "sweep":
            # update config with sweep config
            wandb.init(project="x-perceiver", name=None, resume=True) # init sweep run
            for key, value in wandb.config.items():
                if key in self.config.keys():
                    self.config[key] = value



        train_c_indeces, val_c_indeces, test_c_indeces = [], [], []
        # test_dataloaders = []
        test_data_indices = []
        models = []
        for fold in range(1, self.config["n_folds"]+1):
            print(f"*****FOLD {fold}*****")
            # fix random seeds for reproducibility
            torch.manual_seed(fold)
            np.random.seed(fold)

            train_data, val_data, test_data = self.load_data(fold=fold)
            # get test data indices
            test_data_indices.append(test_data.dataset.indices)
            # test_dataloaders.append(test_data)
            model = self.make_model(train_data)
            wandb.watch(model)
            model, _, train_c_index, _, val_c_index, _, test_c_index = self.train_survival_fold(model, train_data, val_data, test_data, fold=fold)
            train_c_indeces.append(train_c_index)
            val_c_indeces.append(val_c_index)
            test_c_indeces.append(test_c_index)
            models.append(model)

        # log average and standard deviation across folds
        wandb.log({"mean_train_c_index": np.mean(train_c_indeces),
                   "mean_val_c_index": np.mean(val_c_indeces),
                   "std_train_c_index": np.std(train_c_indeces),
                    "std_val_c_index": np.std(val_c_indeces),
                    "mean_test_c_index": np.mean(test_c_indeces),
                    "std_test_c_index": np.std(test_c_indeces)})


        best_fold = np.argmax(test_c_indeces)
        best_model = models[best_fold]

        if self.config.explainer:
            torch.save(best_model.state_dict(), self.log_dir.joinpath("best_model.pt"))
            # save config
            pickle_obj(self.config, self.log_dir.joinpath("config.pkl"))
            # save test data indices
            pickle_obj(test_data_indices[best_fold], self.log_dir.joinpath("test_data_indices.pkl"))

        wandb.finish()

    def load_data(self, fold: int = None) -> tuple:

        level_dict = {
            "blca": 2,
            "brca": 2,
            "kirp": 2,
            "ucec": 2,
            "hnsc": 1,
            "paad": 1,
            "luad": 1,
            "lusc": 1
        }


        data = TCGADataset(self.config["dataset"],
                           self.config,
                           level=level_dict[self.config["dataset"]],
                           # level=int(self.config["data.wsi_level"]),
                           survival_analysis=True,
                           sources=self.sources,
                           n_bins=self.output_dims,
                           log_dir=self.log_dir,
                           )
        train_size = 0.7
        test_size = 0.15
        val_size = 0.15

        print(f"Train samples: {int(train_size*len(data))}, Val samples: {int(val_size * len(data))}, "
              f"Test samples: {int(test_size * len(data))}")
        train, test, val = torch.utils.data.random_split(data, [train_size, test_size, val_size])

        target_distribution = lambda idx, data: dict(np.round(data.omic_df.iloc[idx]["y_disc"].value_counts().sort_values() / len(idx), 2))

        print(f"Train distribution: {target_distribution(train.indices, data)}")
        print(f"Val distribution: {target_distribution(val.indices, data)}")
        print(f"Test distribution: {target_distribution(test.indices, data)}")

        # calculate class weights
        if self.config[f"model_params.class_weights"] == "None":
            self.class_weights = None
        else:
            self.class_weights = torch.Tensor(self._calc_class_weights(train)).to(self.device)

        train_data = DataLoader(train,
                                batch_size=self.config["train_loop.batch_size"],
                                shuffle=True,
                                num_workers=multiprocessing.cpu_count(),
                                pin_memory=True,
                                multiprocessing_context=MP_CONTEXT,
                                persistent_workers=True,
                                prefetch_factor=2
                                )
        val_data = DataLoader(val,
                             batch_size=self.config["train_loop.batch_size"],
                             shuffle=False,
                             num_workers=multiprocessing.cpu_count(),
                             pin_memory=True,
                             multiprocessing_context=MP_CONTEXT,
                             persistent_workers=True,
                             prefetch_factor=2)

        test_data = DataLoader(test,
                               batch_size=self.config["train_loop.batch_size"],
                               shuffle=False,
                               num_workers=multiprocessing.cpu_count(),
                               pin_memory=True,
                               multiprocessing_context=MP_CONTEXT,
                               persistent_workers=True,
                               prefetch_factor=2)




        return train_data, val_data, test_data

    def _calc_class_weights(self, train):

        # if self.config.model in ["healnet", "healnet_early"]:
        if self.config[f"model_params.class_weights"] in ["inverse", "inverse_root"]:
            train_targets = np.array(train.dataset.y_disc)[train.indices]
            _, counts = np.unique(train_targets, return_counts=True)
            if self.config[f"model_params.class_weights"] == "inverse":
                class_weights = 1. / counts
            elif self.config[f"model_params.class_weights"] == "inverse_root":
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
        if self.config.model in  ["healnet", "healnet_early"]:

            num_sources = len(self.config["sources"])
            if num_sources == 1:
                input_channels = [feat[0].shape[2]]
                input_axes = [1]
                modalities = 1
            elif num_sources == 2 and self.config.model == "healnet":
                input_channels = [feat[0].shape[2], feat[1].shape[2]]
                input_axes = [1, 1]
                modalities = 2

            # early fusion healnet (concatenation, so just one modality)
            elif num_sources == 2 and self.config.model == "healnet_early":
                modalities = 1 # same model just single modality
                input_channels = [feat[0].shape[2]]
                input_axes = [1]
            model = HealNet(
                modalities=modalities,
                input_channels=input_channels, # number of features as input channels
                input_axes=input_axes, # second axis (b n_feats c)
                num_classes=self.output_dims,
                num_freq_bands=self.config[f"model_params.num_freq_bands"],
                depth=self.config[f"model_params.depth"],
                max_freq=self.config[f"model_params.max_freq"],
                num_latents = self.config[f"model_params.num_latents"],
                latent_dim = self.config[f"model_params.latent_dim"],
                cross_dim_head = self.config[f"model_params.cross_dim_head"],
                latent_dim_head = self.config[f"model_params.latent_dim_head"],
                cross_heads = self.config[f"model_params.cross_heads"],
                latent_heads = self.config[f"model_params.latent_heads"],
                attn_dropout = self.config[f"model_params.attn_dropout"],
                ff_dropout = self.config[f"model_params.ff_dropout"],
                weight_tie_layers = self.config[f"model_params.weight_tie_layers"],
                fourier_encode_data = self.config[f"model_params.fourier_encode_data"],
                self_per_cross_attn = self.config[f"model_params.self_per_cross_attn"],
                final_classifier_head = True,
                snn = self.config[f"model_params.snn"],
            )
            model.float()
            model.to(self.device)

        elif self.config.model == "fcnn":
            model = RegularizedFCNN(output_dim=self.output_dims)
            model.to(self.device)

        elif self.config.model == "mm_prognosis":
            if len(self.config["sources"]) == 1:
                input_dim = feat[0].shape[1]
                # input_dim = feat[0].shape[2] + feat[1].shape[2]
            model = MMPrognosis(sources=self.sources,
                                output_dims=self.output_dims,
                                config=self.config
                                )
            model.float()
            model.to(self.device)

        elif self.config.model == "mcat":
            if len(self.config["sources"]) == 2:
                model = MCAT(
                    n_classes=self.output_dims,
                    omic_shape=feat[0].shape[1:],
                    wsi_shape=feat[1].shape[1:]
                )
            elif self.config["sources"][0] == "omic":
                model = SNN(
                    n_classes=self.output_dims,
                    input_dim=feat[0].shape[1]
                )
            elif self.config["sources"][0] == "slides":
                model = MILAttentionNet(
                    input_dim=feat[0].shape[1:],
                    n_classes=self.output_dims
                )
            model.float()
            model.to(self.device)

        return model


    def train_survival_fold(self,
                            model: nn.Module,
                            train_data: DataLoader,
                            test_data: DataLoader,
                            val_data: DataLoader,
                            fold: int,
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


        early_stopping = EarlyStopping(patience=self.config["train_loop.patience"],
                                       mode="min", verbose=True)
                                       # delta=self.config["train_loop.delta"],
                                       # maximize=True) # val_c_index as stopping criterion

        model.train()


        for epoch in range(1, self.config["train_loop.epochs"]+1):
            print(f"Epoch {epoch}")
            risk_scores = []
            censorships = []
            event_times = []
            train_loss_surv, train_loss = 0.0, 0.0 # train_loss includes regularisation, train_loss_surve doesn't and is used for logging
            # train_loss = 0.0

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

                dataset = self.config.dataset
                reg_loss = calc_reg_loss(model, self.config[f"model_params.l1"], self.config.model, self.config.sources)

                # log risk, censorship and event time for concordance index
                risk_scores.append(risk)
                censorships.append(censorship.detach().cpu().numpy())
                event_times.append(event_time.detach().cpu().numpy())

                loss_value = loss.item()
                train_loss_surv += loss_value
                train_loss += loss_value + reg_loss
                # backward pass
                loss = loss / gc + reg_loss # gradient accumulation step
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            train_loss /= len(train_data)
            train_loss_surv /= len(train_data)

            risk_scores_full = np.concatenate(risk_scores)
            censorships_full = np.concatenate(censorships)
            event_times_full = np.concatenate(event_times)

            # calculate epoch-level concordance index
            train_c_index = concordance_index_censored((1-censorships_full).astype(bool), event_times_full, risk_scores_full, tied_tol=1e-08)[0]
            wandb.log({f"fold_{fold}_train_loss": train_loss_surv, f"fold_{fold}_train_c_index": train_c_index}, step=epoch if fold == 1 else None)
            print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_c_index))



            # evaluate at interval or if final epoch
            # if epoch % self.config["train_loop.eval_interval"] == 0 or epoch == self.config["train_loop.epochs"]:
            print(f"Running validation")
            val_loss, val_c_index = self.evaluate_survival_epoch(epoch, model, val_data)
            print('Epoch: {}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss, val_c_index))
            wandb.log({f"fold_{fold}_val_loss": val_loss, f"fold_{fold}_val_c_index": val_c_index}, step=epoch if fold == 1 else None)

            if self.config["train_loop.early_stopping"] and early_stopping.step(val_loss, model):
                print(f"Early stopping at epoch {epoch}")
                model = early_stopping.load_best_weights(model)
                break

        # once stopped and best model is loaded, evaluate on test set
        print(f"Running test set evaluation")
        test_loss, test_c_index = self.evaluate_survival_epoch(epoch, model, test_data)
        print('Epoch: {}, test_loss: {:.4f}, test_c_index: {:.4f}'.format(epoch, test_loss, test_c_index))
        wandb.log({f"fold_{fold}_test_loss": test_loss, f"fold_{fold}_test_c_index": test_c_index}, step=epoch if fold == 1 else None)

        # return values of final epoch
        return model, train_loss, train_c_index, val_loss, val_c_index, test_loss, test_c_index

            # checkpoint model after epoch
            # if epoch % self.config["train_loop.checkpoint_interval"] == 0:
            #     torch.save(model.state_dict(), f"{self.log_path}/model_epoch_{epoch}.pt")

    def evaluate_survival_epoch(self,
                                epoch: int,
                                model: nn.Module,
                                test_data: DataLoader,
                                # loss_reg: float=0.0,
                                **kwargs):

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

            logits = model.forward(features)
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1-hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()

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

            reg_loss = calc_reg_loss(model, self.config[f"model_params.l1"], self.config.model, self.config.sources)

            # log risk, censorship and event time for concordance index
            risk_scores.append(risk)
            censorships.append(censorship.detach().cpu().numpy())
            event_times.append(event_time.detach().cpu().numpy())

            loss_value = loss.item()
            val_loss_surv += loss_value
            val_loss += loss_value + reg_loss

            predictions.append(logits.argmax(1).cpu().tolist())
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

        model.train()
        # return unregularised loss for logging
        return val_loss_surv, val_c_index


# def run_plan_iter(args):
#     iteration, params, config, cl_args = args
#     n_folds = 3
#     dataset, sources, model = params["dataset"], params["sources"], params["model"]
#     # print(f"Run plan iteration {iteration+1}/{len(grid)}")
#     print(f"New run plan iteration")
#     print(f"Dataset: {dataset}, Sources: {sources}, Model: {model}")
#
#     # skip healnet_early on single modality (same as regular healnet)
#     if model == "healnet_early" and len(sources) == 1:
#         return None
#     config["dataset"] = dataset
#     config["sources"] = sources
#     config["model"] = model
#     config["n_folds"] = n_folds
#     pipeline = Pipeline(
#             config=config,
#             args=cl_args,
#         )
#     pipeline.main()





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run main training pipeline of x-perceiver")

    # assumes execution
    parser.add_argument("--config_path", type=str, default="/home/kh701/pycharm/x-perceiver/config/main_gpu.yml", help="Path to config file")
    parser.add_argument("--mode", type=str, default="single_run", choices=["single_run", "sweep", "run_plan", "reg_ablation"])
    parser.add_argument("--sweep_config", type=str, default="config/sweep_bayesian.yaml", help="Hyperparameter sweep configuration")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset for run plan")
    parser.add_argument("--datasets", type=list, default=["blca", "brca", "ucec", "kirp"], help="Datasets for run plan")

    # call config
    args = parser.parse_args()
    MP_CONTEXT = "fork"
    # set up multiprocessing context for PyTorch
    torch.multiprocessing.set_start_method(MP_CONTEXT)
    config_path = args.config_path
    config = Config(config_path).read()
    if args.dataset is not None: # for command line sweeps
        config["dataset"] = args.dataset
    # get best hyperparameters for dataset
    hyperparams = Config(config["hyperparams"]).read()[config.dataset]
    config["model_params"] = hyperparams

    if args.mode == "run_plan":
        if args.dataset is not None:
            datasets = [args.dataset]
        else:
            datasets = args.datasets

        grid = ParameterGrid(
            {"dataset": datasets,
             "sources": [["omic", "slides"], ["omic"], ["slides"]],
             "model": ["healnet", "healnet_early"]
             })

        n_folds = 5

        for iteration, params in enumerate(grid):
            dataset, sources, model = params["dataset"], params["sources"], params["model"]
            print(f"Run plan iteration {iteration+1}/{len(grid)}")
            print(f"Dataset: {dataset}, Sources: {sources}, Model: {model}")

            # skip healnet_early on single modality (same as regular healnet)
            if model == "healnet_early" and len(sources) == 1:
                continue
            config["dataset"] = dataset
            config["sources"] = sources
            config["model"] = model
            config["n_folds"] = n_folds
            try:
                pipeline = Pipeline(
                        config=config,
                        args=args,
                    )
                pipeline.main()
            except Exception as e:
                print(f"Exception: {e}")
                continue

        print(f"Successfully finished runplan: "
              f"{list(grid)}")

    elif args.mode == "reg_ablation":
        config["dataset"] = "kirp"
        config["sources"] = ["omic"]
        config["model"] = "healnet"
        config["n_folds"] = 1
        config["train_loop.early_stopping"] = False
        config["train_loop.epochs"] = 50
        regs = [0, 0.00025, 0.00045]
        snn = [True]

        for reg in regs:
            for s in snn:
                config["model_params.l1"] = reg
                config["model_params.snn"] = s
                pipeline = Pipeline(
                        config=config,
                        args=args,
                    )
                pipeline.main()


    else: # single_run or sweep
        pipeline = Pipeline(
                    config=config,
                    args=args,
                )
        pipeline.main()