# State
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
import torch
from torch import nn, Tensor
from itertools import cycle


class InitState(nn.Module, ABC):
    """Trainable initial state"""

    def __init__(self, state_size: int):
        super().__init__()
        self.state_size = state_size

    @abstractmethod
    def forward(self, batch_size) -> Tensor:
        pass


class TrainableInitState(InitState):
    """Trainable initial state"""

    def __init__(self, state_size: int, device: Optional[torch.device] = None):
        super().__init__(state_size)
        self.device = device
        self.state_value = nn.Parameter(
            torch.randn((1, state_size), requires_grad=True, device=self.device)
        )

    def forward(self, batch_size) -> Tensor:
        init_tensor = torch.tile(self.state_value, [batch_size, 1])

        return init_tensor


class MultiModNHistory():
    """Training history of MultiModN"""

    def __init__(
        self,
        targets: List[str]
    ):
        self.decoder_names: List[str] = targets

        self.state_change_loss: List[np.ndarray] = []

        self.loss: Dict[str, List[np.ndarray]] = {
            'train': [],
        }
        self.accuracy: Dict[str, List[np.ndarray]] = {
            'train': [],
        }

        self.sensitivity: Dict[str, List[np.ndarray]] = {
            'train': [],
        }
        self.specificity: Dict[str, List[np.ndarray]] = {
            'train': [],
        }
        self.balanced_accuracy: Dict[str, List[np.ndarray]] = {
            'train': [],
        }

    def plot(
        self,
        filepath: str,
        targets_to_display: List[str],
        show_state_change: bool = False,
    ):
        n_cols = len(self.loss)
        n_rows = 5 # number of metrics to plot
        fig, ax = plt.subplots(figsize=(10*n_cols, 5*n_rows), nrows=n_rows, ncols=n_cols)

        # Plot state change loss curve
        if show_state_change:
            ax.plot(list(map(lambda x: x[-1], self.state_change_loss)), label='State change loss')

        # Plot curves for each target
        for i, target_name in enumerate(targets_to_display):
            if target_name not in self.decoder_names:
                raise ValueError(
                    f"Target name '{target_name}' is not part of the MultiModN history"
                )

            # Plot loss curves
            for col_idx, (key, value) in enumerate(self.loss.items()):
                label = f"{target_name}"
                ax[0][col_idx].plot(list(map(lambda x: x[-1][i], value)), label=label)
                ax[0][col_idx].legend(loc="best")
                ax[0][col_idx].set_title(f"{key.capitalize()} Loss")
                ax[0][col_idx].grid(True)
            # Plot accuracy curves
            for col_idx, (key, value) in enumerate(self.accuracy.items()):
                label = f"{target_name}"
                ax[1][col_idx].plot(list(map(lambda x: x[-1][i], value)), label=label)
                ax[1][col_idx].legend(loc="best")
                ax[1][col_idx].set_title(f"{key.capitalize()} Accuracy")
                ax[1][col_idx].grid(True)

            # Should be similar to accuracy and loss, but I don't use this function, so I hadn't implement it yet
            # Plot sensitivity curves
            for col_idx, (key, value) in enumerate(self.sensitivity.items()):
                label = f"{target_name}"
                ax[2][col_idx].plot(list(map(lambda x: x[-1][i], value)), label=label)
                ax[2][col_idx].legend(loc="best")
                ax[2][col_idx].set_title(f"{key.capitalize()} Sensitivity")
                ax[2][col_idx].grid(True)

            # Plot specificity curves
            for col_idx, (key, value) in enumerate(self.specificity.items()):
                label = f"{target_name}"
                ax[3][col_idx].plot(list(map(lambda x: x[-1][i], value)), label=label)
                ax[3][col_idx].legend(loc="best")
                ax[3][col_idx].set_title(f"{key.capitalize()} Specificity")
                ax[3][col_idx].grid(True)

            # Plot balanced accuracy curves
            for col_idx, (key, value) in enumerate(self.balanced_accuracy.items()):
                label = f"{target_name}"
                ax[4][col_idx].plot(list(map(lambda x: x[-1][i], value)), label=label)
                ax[4][col_idx].legend(loc="best")
                ax[4][col_idx].set_title(f"{key.capitalize()} Balanced Accuracy")
                ax[4][col_idx].grid(True)

        plt.tight_layout()
        fig.savefig(filepath)

    def get_results(self) -> pd.DataFrame:
        n_metrics = len(self.loss) + len(self.accuracy) + len(self.sensitivity) +\
                    len(self.specificity) + len(self.balanced_accuracy) + 1
        start_index = 0
        columns = []
        results = np.zeros((len(self.decoder_names), n_metrics))

        # State change loss, same value for each row (independent of decoder)
        results[:, start_index] = [self.state_change_loss[-1][-1]]*len(self.decoder_names)
        columns.append('State change loss')
        start_index += 1

        # Losses
        for j, (key, value) in enumerate(self.loss.items()):
            columns.append(f'{display_title(key)} loss')

            for i, _ in enumerate(self.decoder_names):
                results[i, j+start_index] = value[-1][-1][i]
        start_index += len(self.loss)

        # Accuracies
        for j, (key, value) in enumerate(self.accuracy.items()):
            columns.append(f'{display_title(key)} accuracy')

            for i, _ in enumerate(self.decoder_names):
                results[i, j+start_index] = value[-1][-1][i]
        start_index += len(self.accuracy)

        # Sensitivities
        for j, (key, value) in enumerate(self.sensitivity.items()):
            columns.append(f'{display_title(key)} sensitivity')

            for i, _ in enumerate(self.decoder_names):
                results[i, j+start_index] = value[-1][-1][i]
        start_index += len(self.sensitivity)

        # Specificities
        for j, (key, value) in enumerate(self.specificity.items()):
            columns.append(f'{display_title(key)} specificity')

            for i, _ in enumerate(self.decoder_names):
                results[i, j+start_index] = value[-1][-1][i]
        start_index += len(self.specificity)

        # Balances accuracies
        for j, (key, value) in enumerate(self.balanced_accuracy.items()):
            columns.append(f'{display_title(key)} balanced accuracy')

            for i, _ in enumerate(self.decoder_names):
                results[i, j+start_index] = value[-1][-1][i]
        start_index += len(self.balanced_accuracy)

        df = pd.DataFrame(results, columns=columns)
        df.index = self.decoder_names

        return df

    def print_results(self):
        print(self.get_results())

    def save_results(self, path):
        df = self.get_results()

        df.to_csv(path, index_label='Target')


def display_title(key: str):
    title = key.replace('_', ' ')
    title = title.capitalize()

    return title
