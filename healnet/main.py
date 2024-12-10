import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, ParameterGrid
import multiprocessing
import argparse
from argparse import Namespace
import yaml
from tqdm import tqdm

from healnet.utils import EarlyStopping, calc_reg_loss, pickle_obj
from healnet.models.survival_loss import CrossEntropySurvLoss, CoxPHSurvLoss, nll_loss
from healnet.baselines import RegularizedFCNN, MMPrognosis, MCAT, SNN, MILAttentionNet, MultiModNModule
from healnet.baselines.multimodn import MLPEncoder, PatchEncoder, ClassDecoder
from healnet.models import HealNet
from healnet.utils import Config, flatten_config
from healnet.etl import TCGADataset

from torch.utils.data import Dataset, DataLoader
import numpy as np
from sksurv.metrics import concordance_index_censored
from torch import optim
import pandas as pd
from box import Box
from pathlib import Path
from datetime import datetime
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)
import wandb



class Pipeline:
    """
    Main experimental pipeline class for training and evaluating models, config handling, and logging
    """

    def __init__(self, config: Box, args: Namespace, wandb_name: str=None):
        self.config = flatten_config(config)
        self.dataset = self.config.dataset
        self.args = args
        self.log_dir = None
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

            sweep_id = wandb.sweep(sweep=sweep_config, project="healnet")
            wandb.agent(sweep_id, function=self.main)
        else:
            wandb_config = dict(self.config)
            wandb.init(project="healnet", name=self.wandb_name, config=wandb_config, resume=True)
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

        valid_models = ["healnet", "fcnn", "healnet_early", "mcat", "mm_prognosis", "multimodn"]
        assert self.config.model in valid_models, f"Invalid model specified. Valid models are {valid_models}"

        valid_class_weights = ["inverse", "inverse_root", "None"]
        assert self.config[f"model_params.class_weights"] in valid_class_weights, f"Invalid class weight specified. " \
                                                                                f"Valid weights are {valid_class_weights}"

        return None


    def main(self):

        # Initialise wandb run (do here for sweep)
        if self.args.mode == "sweep":
            # update config with sweep config
            wandb.init(project="healnet", name=None, resume=True) # init sweep run
            for key, value in wandb.config.items():
                if key in self.config.keys():
                    self.config[key] = value



        train_c_indeces, val_c_indeces, test_c_indeces = [], [], []
        # test_dataloaders = []
        test_data_indices = []
        missing_perfs = []
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
            model, _, train_c_index, _, val_c_index, _, test_c_index, missing_performance = self.train_survival_fold(model, train_data, val_data, test_data, fold=fold)
            train_c_indeces.append(train_c_index)
            val_c_indeces.append(val_c_index)
            test_c_indeces.append(test_c_index)
            missing_perfs.append(missing_performance)
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

        # if missing, log also that
        if self.config.missing_ablation:
            missing_50_c_index, missing_omic_c_index, missing_wsi_c_index = np.mean(missing_perfs, axis=0)
            wandb.log({"missing_50_c_index": missing_50_c_index,
                       "missing_omic_c_index": missing_omic_c_index,
                       "missing_wsi_c_index": missing_wsi_c_index})


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
            "hnsc": 2,
            "paad": 2,
            "luad": 2,
            "lusc": 2
        }


        data = TCGADataset(self.config["dataset"],
                           self.config,
                           level=int(self.config["data.wsi_level"]),
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
                                # num_workers=int(multiprocessing.cpu_count()),
                                num_workers=8,
                                pin_memory=True,
                                multiprocessing_context=MP_CONTEXT,
                                persistent_workers=True,
                                prefetch_factor=2
                                )
        val_data = DataLoader(val,
                             batch_size=self.config["train_loop.batch_size"],
                             shuffle=False,
                             num_workers=int(multiprocessing.cpu_count()),
                             pin_memory=True,
                             multiprocessing_context=MP_CONTEXT,
                             persistent_workers=True,
                             prefetch_factor=2)

        test_data = DataLoader(test,
                               batch_size=self.config["train_loop.batch_size"],
                               shuffle=False,
                               num_workers=int(multiprocessing.cpu_count()),
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
                input_axes = [1, 1] # one axis (MIL and tabular)
                modalities = 2

            # early fusion healnet (concatenation, so just one modality)
            elif num_sources == 2 and self.config.model == "healnet_early":
                modalities = 1 # same model just single modality
                input_channels = [feat[0].shape[2]]
                input_axes = [1]
            model = HealNet(
                n_modalities=modalities,
                channel_dims=input_channels, # number of features as input channels
                num_spatial_axes=input_axes, # second axis (b n_feats channels)
                out_dims=self.output_dims,
                num_freq_bands=self.config[f"model_params.num_freq_bands"],
                depth=self.config[f"model_params.depth"],
                max_freq=self.config[f"model_params.max_freq"],
                l_c = self.config[f"model_params.num_latents"],
                l_d = self.config[f"model_params.latent_dim"],
                cross_dim_head = self.config[f"model_params.cross_dim_head"],
                latent_dim_head = self.config[f"model_params.latent_dim_head"],
                x_heads = self.config[f"model_params.cross_heads"],
                l_heads = self.config[f"model_params.latent_heads"],
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

        elif self.config.model == "multimodn":
            l_d = 2000
            tab_features = feat[0].shape[1]
            patch_dims = feat[1].shape[2]
            encoders = [
                MLPEncoder(state_size=l_d, hidden_layers=[1024, 256, 128, 64], n_features=tab_features),
                PatchEncoder(state_size=l_d, hidden_layers=[512, 256, 128, 64], n_features=patch_dims)
            ]
            decoders = [ClassDecoder(state_size=l_d, n_classes=self.output_dims, activation=torch.sigmoid)]


            model = MultiModNModule(
                state_size=l_d,
                encoders=encoders,
                decoders=decoders
            )
            model.float()
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
            model (nn.Module): model to train
            train_data (DataLoader): training data
            test_data (DataLoader): test data
            **kwargs:

        Returns:
            Tuple: tuple of the model and all performance metrics for given fold
        """
        print(f"Training survival model using {self.config.model}")
        optimizer = optim.Adam(model.parameters(), lr=self.config["optimizer.lr"])
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
            grad_norms = []

            for batch, (features, censorship, event_time, y_disc) in enumerate(tqdm(train_data)):
                # only move to GPU now (use CPU for preprocessing)
                features = [feat.to(self.device) for feat in features] # features available for patient
                censorship = censorship.to(self.device) # status 0 or 1
                event_time = event_time.to(self.device) # survival months (continuous)
                y_disc = y_disc.to(self.device) # discretized survival time bucket

                if batch == 0 and epoch == 1: # print model summary
                    print(f"Modality shapes: ")
                    [print(feat.shape) for feat in features]
                    print(f"Modality dtypes:")
                    [print(feat.dtype) for feat in features]

                optimizer.zero_grad()
                # forward + backward + optimize
                if self.config["model"] == "multimodn":
                    # note that we need to pass the target here for the intermediate loss calc
                    model_loss, logits = model.forward(features, F.one_hot(y_disc, num_classes=self.output_dims))
                else:
                    logits = model.forward(features)
                    model_loss = 0.0
                y_hat = torch.topk(logits, k=1, dim=1)[1]
                hazards = torch.sigmoid(logits)  # sigmoid to get hazards from predictions for surv analysis
                survival = torch.cumprod(1-hazards, dim=1)  # as per paper, survival = cumprod(1-hazards)
                risk = -torch.sum(survival, dim=1).detach().cpu().numpy()  # risk = -sum(survival)

                if self.config["survival.loss"] == "nll":
                    # loss_fn = NLLSurvLoss()
                    # loss = loss_fn(h=hazards, y=y_disc, c=censorship)
                    surv_loss = nll_loss(hazards=hazards, S=survival, Y=y_disc, c=censorship, weights=self.class_weights)
                elif self.config["survival.loss"] == "ce_survival":
                    loss_fn = CrossEntropySurvLoss()
                    surv_loss = loss_fn(hazards=hazards, survival=survival, y_disc=y_disc, censorship=censorship)
                elif self.config["survival.loss"] == "cox":
                    loss_fn = CoxPHSurvLoss()
                    loss_fn(hazards=hazards, survival=survival, censorship=censorship)


                dataset = self.config.dataset
                reg_loss = calc_reg_loss(model, self.config[f"model_params.l1"], self.config.model, self.config.sources)

                # log risk, censorship and event time for concordance index
                risk_scores.append(risk)
                censorships.append(censorship.detach().cpu().numpy())
                event_times.append(event_time.detach().cpu().numpy())

                loss_value = surv_loss.item()
                train_loss_surv += loss_value
                train_loss += loss_value + reg_loss
                # backward pass
                surv_loss = surv_loss / gc + reg_loss + model_loss  # gradient accumulation step
                surv_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

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

        # run ablation
        missing_performance = None
        if self.config.missing_ablation:
            _, missing_50_c_index = self.evaluate_survival_epoch(epoch=self.config["train_loop.epochs"],
                                                                               model=model,
                                                                               test_data=test_data,
                                                                               missing_mode="50")
            _, missing_omic_c_index = self.evaluate_survival_epoch(epoch=self.config["train_loop.epochs"],
                                                                               model=model,
                                                                               test_data=test_data,
                                                                               missing_mode="omic")
            _, missing_wsi_c_index = self.evaluate_survival_epoch(epoch=self.config["train_loop.epochs"],
                                                                               model=model,
                                                                               test_data=test_data,
                                                                               missing_mode="wsi")

            missing_performance = (missing_50_c_index, missing_omic_c_index, missing_wsi_c_index)



        # return values of final epoch
        return model, train_loss, train_c_index, val_loss, val_c_index, test_loss, test_c_index, missing_performance

    def _sample_missing(self, features, use_omic, mode):
        assert mode in ["50", "omic", "wsi"], "Invalid missing ablation mode"

        if mode == "50":
            if use_omic:
                use_omic = False
                return [features[0]], use_omic
            else:
                use_omic = True
                return [features[1]], use_omic
        elif mode == "omic":
            # return only WSIs
            return [features[1]], None
        elif mode == "wsi":
            # return only omic
            return [features[0]], None


    def evaluate_survival_epoch(self,
                                epoch: int,
                                model: nn.Module,
                                test_data: DataLoader,
                                missing_mode: str=None,
                                # loss_reg: float=0.0,
                                **kwargs):

        model.eval()
        risk_scores = []
        censorships = []
        event_times = []
        predictions = []
        labels = []
        val_loss_surv, val_loss = 0.0, 0.0
        use_omic = True

        for batch, (features, censorship, event_time, y_disc) in enumerate(tqdm(test_data)):
            # only move to GPU now (use CPU for preprocessing)
            if missing_mode is not None: # handle for missing modality ablation
                features, use_omic = self._sample_missing(features, use_omic, missing_mode)
            features = [feat.to(self.device) for feat in features]
            censorship = censorship.to(self.device)
            event_time = event_time.to(self.device)
            y_disc = y_disc.to(self.device)

            if self.config["model"] == "multimodn":
                model_loss, logits = model.forward(features, F.one_hot(y_disc, num_classes=self.output_dims))
            else:
                logits = model.forward(features)
                model_loss = 0.0
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
            val_loss += loss_value + reg_loss + model_loss

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

    def calc_gradient_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run main training pipeline of healnet")

    # assumes execution
    parser.add_argument("--config_path", type=str, default="config/main_gpu.yml", help="Path to config file")
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
    # get best hyperparameters for datKaset
    hyperparams = Config(config["hyperparams"]).read()[config.dataset]
    config["model_params"] = hyperparams

    if args.mode == "run_plan":
        if args.dataset is not None:
            datasets = [args.dataset]
        else:
            datasets = args.datasets

        grid = ParameterGrid(
            {"dataset": datasets,
             "sources": [["omic", "slides"]],
             "model": ["healnet"]
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
        config["sources"] = ["omic", "slides"]
        config["model"] = "healnet"
        config["n_folds"] = 1
        config["train_loop.early_stopping"] = False
        config["train_loop.epochs"] = 50
        regs  = [2.0,1.0]
        snn = [True, False] # False
        sets = ["blca", "brca", "ucec", "kirp"]

        for dataset in sets:
            config["dataset"] = dataset
            best_reg = config["model_params.l1"]
            for reg in regs:
                config["model_params.l1"] = best_reg / reg
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