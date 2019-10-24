import numpy as np
import time
import torch
import torch.nn as nn
from pytorch_model.mlp import Mlp, Linear
from pytorch_model.ops import get_optimizer
from pytorch_model.ops import evaluate
from pytorch_model.ops import str_to_metric


class ClassifierBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.optimizer = None

    def initialize(self):
        pass

    def forward(self, *input):
        pass

    def clear_gradient(self):
        if isinstance(self.optimizer, list):
            for op in self.optimizer:
                op.zero_grad()
        else:
            self.optimizer.zero_grad()

    def step_gradient(self):
        if isinstance(self.optimizer, list):
            for op in self.optimizer:
                op.step()
        else:
            self.optimizer.step()

    def fit_one_step(self, train_loader, grad_norm_clip=None, device=None, eval_metrics=None,
                     log_freq=None):
        self.train()
        start_time = time.time()
        sum_eval = None
        sum_loss = 0.
        if eval_metrics is not None:
            sum_eval = [0.] * len(eval_metrics)
        for batch, sample in enumerate(train_loader, start=1):
            self.clear_gradient()
            for k, v in sample.items():
                sample[k] = v.to(device)
            pred = self.forward(sample)
            loss = self.loss_func(pred, sample['label'])
            loss.backward()
            if eval_metrics is not None:
                for i in range(len(eval_metrics)):
                    sum_eval[i] += eval_metrics[i](
                        sample['label'].cpu().detach().numpy(),
                        pred.cpu().detach().numpy()
                    )
            sum_loss += loss.cpu().detach().numpy()
            if grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_norm_clip)
            self.step_gradient()
            if log_freq is not None and batch % log_freq == 0:
                info = 'batch info {}/{} : loss_ave {:.6f} time/batch = {:.1f}mins'
                print(info.format(
                    batch,
                    len(train_loader),
                    sum_loss / batch,
                    (time.time() - start_time) / 60.
                ))
                if eval_metrics is not None:
                    info2 = 'eval average -->'
                    for _e, _v in zip(eval_metrics, sum_eval):
                        info2 += ' {}: {:.6f}'.format(_e.__name__, _v / batch)
                    print(info2)

    def fit(self, train_dataset, train_loader, valid_dataset, valid_loader,
            num_round, grad_norm_clip=None, device=None, eval_metrics=None, log_freq=None):
        self.to(device)
        train_data = train_loader(train_dataset)
        if eval_metrics is not None:
            if not isinstance(eval_metrics, list):
                eval_metrics = [eval_metrics]
            for i in range(len(eval_metrics)):
                if isinstance(eval_metrics[i], str):
                    eval_metrics[i] = str_to_metric(eval_metrics[i])
        for ite in range(num_round):
            self.fit_one_step(train_data, grad_norm_clip, device, eval_metrics, log_freq)
            if valid_dataset is not None and eval_metrics is not None:
                info = 'evaluate at iteration {}'.format(ite)
                y_pred, y_true = self.predict(valid_dataset, valid_loader, True, device)
                for _e in eval_metrics:
                    info += ' {}: {:.6f}'.format(_e.__name__, _e(y_pred, y_true))
                print(info)

    def predict(self, dataset, data_loader=None, with_label=False, device=None):
        self.to(device)
        self.eval()
        data = data_loader(dataset) if data_loader is not None else dataset
        preds = list()
        labels = list() if with_label else None
        for sample in data:
            for k, v in sample.items():
                sample[k] = v.to(device)
            preds.append(self.forward(sample).cpu().detach().numpy())
            if with_label:
                labels.append(sample['label'].cpu().detach().numpy())
        preds = np.concatenate(preds)
        if with_label:
            labels = np.concatenate(labels)
            return preds, labels
        return preds


class MlpClassifier(ClassifierBase):
    def __init__(self, num_class, in_features, hidden_units):
        super().__init__()
        self.num_class = num_class
        self.in_features = in_features
        self.hidden_units = hidden_units
        self.mlp, self.linear, self.optimizer = None, None, None
        self.initialize()

    def initialize(self):
        self.mlp = Mlp(
            self.in_features,
            self.hidden_units,
            'relu'
        )
        self.linear = Linear(
            self.hidden_units[-1],
            self.num_class,
            'sigmoid' if self.num_class == 1 else 'softmax'
        )
        self.optimizer = get_optimizer('adam', self.parameters(), {'lr': 1e-2})
        self.loss_func = nn.BCELoss()

    def forward(self, data: dict) -> torch.FloatTensor:
        x = data['x']
        x = self.mlp(x)
        x = self.linear(x)
        if x.shape[1] == 1:
            x = x.reshape(-1)
        return x
