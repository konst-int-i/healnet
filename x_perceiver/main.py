import torch
import torch.nn as nn
from tqdm import tqdm
from x_perceiver.train import majority_classifier_acc
from x_perceiver.models import NLLSurvLoss
import numpy as np
from torchsummary import summary
from sksurv.metrics import concordance_index_censored
from torch import optim
from x_perceiver.models.perceiver import Perceiver
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from x_perceiver.utils import Config
from x_perceiver.etl import TCGADataset
from pathlib import Path
from datetime import datetime
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)

class Pipeline:

    def __init__(self, config_path, sources, survival_analysis: bool=True):
        self.config = Config(config_path).read()
        self.output_dims = int(self.config.model.output_dims)
        valid_sources = ["omic", "slides"]
        assert all([source in valid_sources for source in sources]), f"Invalid source specified. Valid sources are {valid_sources}"
        self.sources = sources
        self.survival_analysis = survival_analysis
        # initialise cuda device (will load directly to GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def main(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = Path(self.config.log_path).joinpath(timestamp)
        self.log_path.mkdir(parents=True, exist_ok=True)
        print(f"Logging to {self.log_path}")

        train_data, test_data = self.load_data()

        model = self.make_model(train_data)

        self.train_survival(model, train_data, test_data)



    def load_data(self):
        data = TCGADataset("blca",
                           self.config,
                           level=3,
                           survival_analysis=True,
                           sources=self.sources,
                           n_bins=self.output_dims,
                           )

        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        print(f"Train samples: {train_size}, Test samples: {test_size}")
        train, test = torch.utils.data.random_split(data, [train_size, test_size])
        torch.multiprocessing.set_start_method('spawn')

        train_data = DataLoader(train, batch_size=self.config.model.batch_size, shuffle=False, num_workers=os.cpu_count(),
                                pin_memory=True, multiprocessing_context="spawn")

        test_data = DataLoader(test, batch_size=self.config.model.batch_size, shuffle=False, num_workers=os.cpu_count(),
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
                max_freq=10.,
                num_classes=self.output_dims, # survival analysis expecting n_bins as output dims
                num_latents = 512,
                latent_dim = 512,
            #     num_latents = 16,
            #     latent_dim = 16,
                cross_dim_head = 64,
                latent_dim_head = 64,
                # cross_dim_head = 8,
                # latent_dim_head = 8,
                cross_heads = 1,
                latent_heads = 8,
                attn_dropout = 0.5,  # non-default
                ff_dropout = 0.5,  # non-default
                weight_tie_layers = False,
                fourier_encode_data = True,
                self_per_cross_attn = 1,
                final_classifier_head = True
            )
            perceiver.float()
            perceiver.to(self.device)
        elif self.sources == ["slides"]:
            perceiver = Perceiver(
                input_channels=4,
                input_axis=2,
                num_freq_bands=6,
                depth=1,  # number of cross-attention iterations
                max_freq=10.,
                num_classes=self.output_dims,
                # num_classes=len(set(train_data.dataset.dataset.target)),
            )
            perceiver.to(self.device) # need to move to GPU to get summary
        # set model precision
        return perceiver

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
        # model.to(self.device)
        optimizer = optim.SGD(model.parameters(),
                              lr=self.config.model.lr, momentum=self.config.model.momentum)
        # set efficient OneCycle scheduler, significantly reduces required training iters
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                  max_lr=self.config.model.max_lr,
                                                  epochs=self.config.model.epochs,
                                                  steps_per_epoch=len(train_data))

        criterion = NLLSurvLoss()
        model.train()
        eval_interval = 1


        # majority_train_acc = majority_classifier_acc(train_data.dataset.dataset.target)
        for epoch in range(self.config.model.epochs):
            print(f"Epoch {epoch}")
            risk_scores = []
            censorships = []
            event_times = []
            train_loss_surv, train_loss = 0.0, 0.0

            for batch, (features, censorship, event_time, y_disc) in enumerate(tqdm(train_data)):
                # only move to GPU now (use CPU for preprocessing)
                features = features.to(self.device)
                censorship = censorship.to(self.device)
                event_time = event_time.to(self.device)
                y_disc = y_disc.to(self.device)

                if batch == 0 and epoch == 0: # print model summary
                    print(features.shape)
                    print(features.dtype)

                optimizer.zero_grad()
                # forward + backward + optimize
                y_hat = model.forward(features)
                hazards = torch.sigmoid(y_hat)
                survival = torch.cumprod(1-hazards, dim=1)
                risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
                loss = criterion(h=y_hat, y=y_disc, c=censorship)

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

            print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

            if epoch % eval_interval == 0:
                print("**************************")
                print(f"EPOCH {epoch} EVALUATION")
                print("**************************")
                # self.evaluate_classification(model, test_data, criterion, optimizer)

            # checkpoint model after epoch
            if epoch % self.config.model.checkpoint_interval == 0:
                torch.save(model.state_dict(), f"{self.log_path}/model_epoch_{epoch}.pt")

    def evaluate_survival(self):
        pass

    def evaluate_classification(self, model: nn.Module, test_data: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer):
        model.eval()
        majority_val_acc = majority_classifier_acc(y_true=test_data.dataset.dataset.target)
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for batch, (features, target) in enumerate(test_data):
                features, target = features.to(self.device), target.to(self.device)
                outputs = model.forward(features)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == target).sum().item()
        val_loss = np.round(val_loss / len(test_data), 5)
        val_acc = np.round(val_acc / len(test_data), 5)
        print(f"test_loss: {val_loss}, test_acc: {val_acc}, majority_test_acc: {majority_val_acc}")

        model.train()


if __name__ == "__main__":
    import os
    print(os.getcwd())

    pipeline = Pipeline(
        config_path="/home/kh701/pycharm/x-perceiver/config/main_gpu.yml",
        # sources=["omic"],
        sources=["slides"]
    )
    pipeline.main()

