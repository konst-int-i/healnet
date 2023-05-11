import torch
import torch.nn as nn
from tqdm import tqdm
from x_perceiver.train import majority_classifier_acc
from x_perceiver.models import NLLSurvLoss, CrossEntropySurvLoss, CoxPHSurvLoss
import numpy as np
from torchsummary import summary
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from torch import optim
from x_perceiver.models.perceiver import Perceiver
import pandas as pd
from box import Box
from torch.utils.data import Dataset, DataLoader
from x_perceiver.utils import Config
from x_perceiver.etl import TCGADataset
from pathlib import Path
from datetime import datetime
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)
import wandb



class Pipeline:

    def __init__(self, config: Box, wandb_name: str=None):
        self.config = config
        self.output_dims = int(self.config.model_params.output_dims)
        self.sources = self.config.sources
        self.n_classes = self.config.clf.n_classes
        # initialise cuda device (will load directly to GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            print(f"Setting default cuda tensor to double")
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)

        # fix random seeds for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # setup wandb logging
        wandb_config = {}
        wandb_config.update(dict(self.config))
        wandb.init(
            project="x-perceiver",
            config = wandb_config,
            name=wandb_name,
        )


    def _check_config(self) -> None:
        """
        Assert that the config only contains valid arguments
        Returns:
            None
        """
        valid_sources = ["omic", "slides"]
        assert all([source in valid_sources for source in self.config.sources]), f"Invalid source specified. Valid sources are {valid_sources}"

        valid_survival_losses = ["nll", "ce_survival", "cox"]
        assert self.config.survival.loss in valid_survival_losses, f"Invalid survival loss specified. " \
                                                                   f"Valid losses are {valid_survival_losses}"

        valid_tasks = ["survival", "classification"]
        assert self.config.task in valid_tasks, f"Invalid task specified. Valid tasks are {valid_tasks}"

        return None


    def main(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = Path(self.config.log_path).joinpath(timestamp)
        self.log_path.mkdir(parents=True, exist_ok=True)
        print(f"Logging to {self.log_path}")

        train_data, test_data = self.load_data()
        model = self.make_model(train_data)
        wandb.watch(model)
        if self.config.task == "survival":
            self.train_survival(model, train_data, test_data)

        elif self.config.task == "classification":
            self.train_clf(model, train_data, test_data)

        wandb.finish()

    def load_data(self):
        data = TCGADataset("blca",
                           self.config,
                           level=3,
                           survival_analysis=True,
                           sources=self.sources,
                           n_bins=self.output_dims if self.config.task == "survival" else self.n_classes,
                           )

        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        print(f"Train samples: {train_size}, Test samples: {test_size}")
        train, test = torch.utils.data.random_split(data, [train_size, test_size])

        if self.config.model_params.class_weights:
            # class weighting only for training data
            _, counts = np.unique(train.dataset.y_disc, return_counts=True)
            self.class_weight = 1. / counts
            sample_weights = np.array([self.class_weight[t] for t in train.y_disc])
            sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
        pass_sampler = sampler if self.config.train_loop.use_sampler else None

        train_data = DataLoader(train,
                                batch_size=self.config.train_loop.batch_size,
                                shuffle=False, num_workers=os.cpu_count(),
                                pin_memory=True, multiprocessing_context="spawn", sampler=pass_sampler)

        test_data = DataLoader(test, batch_size=self.config.train_loop.batch_size, shuffle=False, num_workers=os.cpu_count(),
                                pin_memory=True, multiprocessing_context="spawn")
        return train_data, test_data

    def make_model(self, train_data: DataLoader):
        """
        Instantiates model and moves to CUDA device if available
        Args:
            train_data:

        Returns:
            nn.Module: model used for training
        """
        if self.sources == ["omic"]:
            perceiver = Perceiver(
                input_channels=1,
                input_axis=2, # second axis (b n_feats c)
                num_freq_bands=6,
                depth=3,
                max_freq=2.,
                num_classes=self.output_dims, # survival analysis expecting n_bins as output dims
                # num_latents = 512,
                # latent_dim = 512,
                num_latents = 32,
                latent_dim = 32,
                # cross_dim_head = 64,
                # latent_dim_head = 64,
                cross_dim_head = 16,
                latent_dim_head = 16,
                cross_heads = 1,
                latent_heads = 8,
                attn_dropout = 0.5,  # non-default
                ff_dropout = 0.5,  # non-default
                weight_tie_layers = False,
                fourier_encode_data = False,
                self_per_cross_attn = 1,
                final_classifier_head = True
            )
            perceiver.float()
            feat, _, _, _ = next(iter(train_data))
            perceiver.to(self.device)
            summary(perceiver, input_size=feat.shape[1:])
        elif self.sources == ["slides"]:
            perceiver = Perceiver(
                input_channels=3, # RGB, dropped alpha channel
                input_axis=2,
                num_freq_bands=6,
                depth=1,  # number of cross-attention iterations
                max_freq=10.,
                num_classes=self.output_dims,
                num_latents=256,
                latent_dim=256,  # latent dim of transformer
                cross_dim_head=32,
                latent_dim_head=32,
                attn_dropout=0.5,
                ff_dropout=0.5,
                cross_heads=1,
                final_classifier_head=True
            )
            perceiver.to(self.device) # need to move to GPU to get summary
            feat, _, _, _ = next(iter(train_data))
            summary(perceiver, input_size=feat.shape[1:]) # omit batch dim
        # set model precision
        return perceiver


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
                              lr=self.config.optimizer.lr, momentum=self.config.optimizer.momentum)
        # set efficient OneCycle scheduler, significantly reduces required training iters
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                  max_lr=self.config.optimizer.max_lr,
                                                  epochs=self.config.train_loop.epochs,
                                                  steps_per_epoch=len(train_data))
        loss_weight = torch.tensor(self.class_weight).to(self.device) if self.config.model_params.class_weights else None

        criterion = nn.CrossEntropyLoss(weight=loss_weight)
        # use survival loss for survival analysis which accounts for censored data
        model.train()

        majority_train_acc = np.round(majority_classifier_acc(train_data.dataset.dataset.y_disc), 5)
        for epoch in range(self.config.train_loop.epochs):
            print(f"Epoch {epoch}")
            running_loss = 0.0
            # running_acc = 0.0
            predictions = []
            labels = []
            for batch, (features, _, _, y_disc) in enumerate(tqdm(train_data)):
                # only move to GPU now (use CPU for preprocessing)
                labels.append(y_disc.tolist())
                features, y_disc = features.to(self.device), y_disc.to(self.device)
                if batch == 0 and epoch == 0: # print model summary
                    print(features.shape)
                    print(features.dtype)
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model.forward(features)
                loss = criterion(outputs, y_disc)
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
            print(f"Batch {batch}, train_loss: {train_loss}, "
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
            # running_acc = 0.0

            if epoch % self.config.train_loop.eval_interval == 0:
                # print("**************************")
                # print(f"EPOCH {epoch} EVALUATION")
                # print("**************************")
                self.evaluate_clf_epoch(model, test_data, criterion)

            # checkpoint model after epoch
            if epoch % self.config.train_loop.checkpoint_interval == 0:
                torch.save(model.state_dict(), f"{self.log_path}/model_epoch_{epoch}.pt")

    def evaluate_clf_epoch(self, model: nn.Module, test_data: DataLoader, criterion: nn.Module):
        model.eval()
        majority_val_acc = majority_classifier_acc(y_true=test_data.dataset.dataset.y_disc)
        val_loss = 0.0
        val_acc = 0.0
        predictions = []
        labels = []
        with torch.no_grad():
            for batch, (features, _, _, y_disc) in enumerate(test_data):
                labels.append(y_disc.item())
                features, y_disc = features.to(self.device), y_disc.to(self.device)
                outputs = model.forward(features)
                loss = criterion(outputs, y_disc)
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == y_disc).sum().item()
                predictions.append(outputs.argmax(1).cpu().item())
        val_loss = np.round(val_loss / len(test_data), 5)
        # epoch_predictions = np.concatenate(predictions)
        # epoch_labels = np.concatenate(labels)
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
                   # "majority_val_acc": majority_val_acc
                   })
        model.train()


    def train_survival(self,
                       model: nn.Module,
                       train_data: DataLoader,
                       test_data: DataLoader,
                       loss_reg: float = 0.0,
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
        print(f"Training surivival model")
        # model.to(self.device)
        optimizer = optim.SGD(model.parameters(),
                              lr=self.config.optimizer.lr, momentum=self.config.optimizer.momentum)
        # set efficient OneCycle scheduler, significantly reduces required training iters
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                  max_lr=self.config.optimizer.max_lr,
                                                  epochs=self.config.train_loop.epochs,
                                                  steps_per_epoch=len(train_data))

        model.train()
        eval_interval = 10

        for epoch in range(self.config.train_loop.epochs):
            print(f"Epoch {epoch}")
            risk_scores = []
            censorships = []
            event_times = []
            train_loss_surv, train_loss = 0.0, 0.0

            for batch, (features, censorship, event_time, y_disc) in enumerate(tqdm(train_data)):
                # only move to GPU now (use CPU for preprocessing)
                features = features.to(self.device) # features available for patient
                censorship = censorship.to(self.device) # status 0 or 1
                event_time = event_time.to(self.device) # survival months (continuous)
                y_disc = y_disc.to(self.device) # discretized survival time bucket

                if batch == 0 and epoch == 0: # print model summary
                    print(features.shape)
                    print(features.dtype)

                optimizer.zero_grad()
                # forward + backward + optimize
                y_hat = model.forward(features)  # model predictions of survival time bucket (shape is n_bins)
                hazards = torch.sigmoid(y_hat)  # sigmoid to get hazards from predictions for surv analysis
                survival = torch.cumprod(1-hazards, dim=1)  # as per paper, survival = cumprod(1-hazards)
                risk = -torch.sum(survival, dim=1).detach().cpu().numpy()  # risk = -sum(survival)
                if self.config.survival.loss == "nll":
                    loss_fn = NLLSurvLoss()
                    loss = loss_fn(h=y_hat, y=y_disc, c=censorship)
                elif self.config.survival.loss == "ce_survival":
                    loss_fn = CrossEntropySurvLoss()
                    loss = loss_fn(hazards=hazards, survival=survival, y_disc=y_disc, censorship=censorship)
                elif self.config.survival.loss == "cox":
                    loss_fn = CoxPHSurvLoss()
                    loss_fn(hazards=hazards, survival=survival, censorship=censorship)

                # log risk, censorship and event time for concordance index
                risk_scores.append(risk)
                censorships.append(censorship.detach().cpu().numpy())
                event_times.append(event_time.detach().cpu().numpy())

                loss_value = loss.item()
                train_loss_surv += loss_value
                train_loss += loss_value + loss_reg

                # backward pass
                loss = loss / gc + loss_reg # gradient accumulation step
                loss.backward()

                if (batch + 1) % gc == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # calculate epoch-level stats
            train_loss_surv /= len(train_data)
            train_loss /= len(train_data)

            risk_scores_full = np.concatenate(risk_scores)
            censorships_full = np.concatenate(censorships)
            event_times_full = np.concatenate(event_times)

            # calculate epoch-level concordance index
            c_index = concordance_index_censored((1-censorships_full).astype(bool), event_times_full, risk_scores_full)[0]
            wandb.log({"train_loss": train_loss, "train_c_index": c_index}, step=epoch)
            print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss, c_index))


            if epoch % eval_interval == 0:
                val_loss, val_c_index = self.evaluate_survival_epoch(epoch, model, test_data, loss_reg=loss_reg)
                wandb.log({"val_loss": val_loss, "val_c_index": val_c_index}, step=epoch)

            # checkpoint model after epoch
            if epoch % self.config.train_loop.checkpoint_interval == 0:
                torch.save(model.state_dict(), f"{self.log_path}/model_epoch_{epoch}.pt")

    def evaluate_survival_epoch(self,
                                epoch: int,
                                model: nn.Module,
                                test_data: DataLoader,
                                loss_reg: float=0.0,
                                **kwargs):

        # criterion = NLLSurvLoss()
        print(f"Running validation...")
        model.eval()
        risk_scores = []
        censorships = []
        event_times = []
        val_loss_surv, val_loss = 0.0, 0.0

        for batch, (features, censorship, event_time, y_disc) in enumerate(tqdm(test_data)):
            # only move to GPU now (use CPU for preprocessing)
            features = features.to(self.device)
            censorship = censorship.to(self.device)
            event_time = event_time.to(self.device)
            y_disc = y_disc.to(self.device)

            y_hat = model.forward(features)
            hazards = torch.sigmoid(y_hat)
            survival = torch.cumprod(1-hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()

            if self.config.survival.loss == "nll":
                loss_fn = NLLSurvLoss()
                loss = loss_fn(h=y_hat, y=y_disc, c=censorship)
            elif self.config.survival.loss == "ce_survival":
                loss_fn = CrossEntropySurvLoss()
                loss = loss_fn(hazards=hazards, survival=survival, y_disc=y_disc, censorship=censorship)
            elif self.config.survival.loss == "cox":
                loss_fn = CoxPHSurvLoss()
                loss_fn(hazards=hazards, survival=survival, censorship=censorship)

            # log risk, censorship and event time for concordance index
            risk_scores.append(risk)
            censorships.append(censorship.detach().cpu().numpy())
            event_times.append(event_time.detach().cpu().numpy())

            loss_value = loss.item()
            val_loss_surv += loss_value
            val_loss += loss_value + loss_reg


        # calculate epoch-level stats
        val_loss_surv /= len(test_data)
        val_loss /= len(test_data)

        risk_scores_full = np.concatenate(risk_scores)
        censorships_full = np.concatenate(censorships)
        event_times_full = np.concatenate(event_times)

        # calculate epoch-level concordance index
        c_index = concordance_index_censored((1-censorships_full).astype(bool), event_times_full, risk_scores_full)[0]

        print(f"Epoch: {epoch}, val_loss: {val_loss}, val_c_index: {c_index}")

        model.train()
        return val_loss, c_index


if __name__ == "__main__":
    import os
    print(os.getcwd())

    torch.multiprocessing.set_start_method('spawn') #  only set once for repeated experiments to work

    config_path="/home/kh701/pycharm/x-perceiver/config/main_gpu.yml"
    config = Config(config_path).read()
    pipeline = Pipeline(
            config=config,
        )
    pipeline.main()


    # losses = ["nll", "ce_survival"]
    # for loss in losses:
    #     print(loss)
    #     config["train_loop"]["loss"] = loss
    #     display_name = f"omic_30ep_{config['train_loop']['loss']}_loss"
    #     pipeline = Pipeline(
    #         config=config,
    #     )
    #     pipeline.main()