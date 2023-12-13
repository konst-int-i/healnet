"""
Old functions that used to be part of the main pipeline, but are now deprecated.
"""

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