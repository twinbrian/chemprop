from __future__ import annotations

import io
from typing import Iterable
import warnings

from lightning import pytorch as pl
import torch
from torch import Tensor, nn, optim

from chemprop.data import BatchMolGraph, TrainingBatch
from chemprop.nn import Aggregation, ChempropMetric, MessagePassing, Predictor
from chemprop.nn.transforms import ScaleTransform
from chemprop.schedulers import build_NoamLike_LRSched


class MPNN(pl.LightningModule):
    r"""An :class:`MPNN` is a sequence of message passing layers, an aggregation routine, and a
    predictor routine.

    The first two modules calculate learned fingerprints from an input molecule
    reaction graph, and the final module takes these learned fingerprints as input to calculate a
    final prediction. I.e., the following operation:

    .. math::
        \mathtt{MPNN}(\mathcal{G}) =
            \mathtt{predictor}(\mathtt{agg}(\mathtt{message\_passing}(\mathcal{G})))

    The full model is trained end-to-end.

    Parameters
    ----------
    message_passing : MessagePassing
        the message passing block to use to calculate learned fingerprints
    agg : Aggregation
        the aggregation operation to use during molecule-level predictor
    predictor : Predictor
        the function to use to calculate the final prediction
    batch_norm : bool, default=False
        if `True`, apply batch normalization to the output of the aggregation operation
    metrics : Iterable[Metric] | None, default=None
        the metrics to use to evaluate the model during training and evaluation
    warmup_epochs : int, default=2
        the number of epochs to use for the learning rate warmup
    init_lr : int, default=1e-4
        the initial learning rate
    max_lr : float, default=1e-3
        the maximum learning rate
    final_lr : float, default=1e-4
        the final learning rate

    Raises
    ------
    ValueError
        if the output dimension of the message passing block does not match the input dimension of
        the predictor function
    """

    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        batch_norm: bool = False,
        metrics: Iterable[ChempropMetric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
    ):
        super().__init__()
        # manually add X_d_transform to hparams to suppress lightning's warning about double saving
        # its state_dict values.
        self.save_hyperparameters(ignore=["X_d_transform", "message_passing", "agg", "predictor"])
        self.hparams["X_d_transform"] = X_d_transform
        self.hparams.update(
            {
                "message_passing": message_passing.hparams,
                "agg": agg.hparams,
                "predictor": predictor.hparams,
            }
        )

        self.message_passing = message_passing
        self.agg = agg
        self.bn = nn.BatchNorm1d(self.message_passing.output_dim) if batch_norm else nn.Identity()
        self.predictor = predictor

        self.X_d_transform = X_d_transform if X_d_transform is not None else nn.Identity()

        self.metrics = (
            nn.ModuleList([*metrics, self.criterion.clone()])
            if metrics
            else nn.ModuleList([self.predictor._T_default_metric(), self.criterion.clone()])
        )

        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    @property
    def output_dim(self) -> int:
        return self.predictor.output_dim

    @property
    def n_tasks(self) -> int:
        return self.predictor.n_tasks

    @property
    def n_targets(self) -> int:
        return self.predictor.n_targets

    @property
    def criterion(self) -> ChempropMetric:
        return self.predictor.criterion

    def fingerprint(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """the learned fingerprints for the input molecules"""
        H_v = self.message_passing(bmg, V_d)
        H = self.agg(H_v, bmg.batch)
        H = self.bn(H)

        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), 1)

    def encoding(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None, i: int = -1
    ) -> Tensor:
        """Calculate the :attr:`i`-th hidden representation"""
        return self.predictor.encode(self.fingerprint(bmg, V_d, X_d), i)

    def forward(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """Generate predictions for the input molecules/reactions"""
        return self.predictor(self.fingerprint(bmg, V_d, X_d))

    def training_step(self, batch: TrainingBatch, batch_idx):
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor.train_step(Z)
        l = self.criterion(preds, targets, mask, weights, lt_mask, gt_mask)

        self.log("train_loss", self.criterion, prog_bar=True, on_epoch=True)

        return l

    def on_validation_model_eval(self) -> None:
        self.eval()
        self.predictor.output_transform.train()

    def validation_step(self, batch: TrainingBatch, batch_idx: int = 0):
        self._evaluate_batch(batch, "val")

        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor.train_step(Z)
        self.metrics[-1](preds, targets, mask, weights, lt_mask, gt_mask)
        self.log("val_loss", self.metrics[-1], batch_size=len(batch[0]), prog_bar=True)

    def test_step(self, batch: TrainingBatch, batch_idx: int = 0):
        self._evaluate_batch(batch, "test")

    def _evaluate_batch(self, batch: TrainingBatch, label: str) -> None:
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        preds = self(bmg, V_d, X_d)
        weights = torch.ones_like(weights)

        if self.predictor.n_targets > 1:
            preds = preds[..., 0]

        for m in self.metrics[:-1]:
            m.update(preds, targets, mask, weights, lt_mask, gt_mask)
            self.log(f"{label}/{m.alias}", m, batch_size=len(batch[0]))

    def predict_step(self, batch: TrainingBatch, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Return the predictions of the input batch

        Parameters
        ----------
        batch : TrainingBatch
            the input batch

        Returns
        -------
        Tensor
            a tensor of varying shape depending on the task type:

            * regression/binary classification: ``n x (t * s)``, where ``n`` is the number of input
              molecules/reactions, ``t`` is the number of tasks, and ``s`` is the number of targets
              per task. The final dimension is flattened, so that the targets for each task are
              grouped. I.e., the first ``t`` elements are the first target for each task, the second
              ``t`` elements the second target, etc.

            * multiclass classification: ``n x t x c``, where ``c`` is the number of classes
        """
        bmg, X_vd, X_d, *_ = batch

        return self(bmg, X_vd, X_d)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)
        if self.trainer.train_dataloader is None:
            # Loading `train_dataloader` to estimate number of training batches.
            # Using this line of code can pypass the issue of using `num_training_batches` as described [here](https://github.com/Lightning-AI/pytorch-lightning/issues/16060).
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            warnings.warn(
                "For infinite training, the number of cooldown epochs in learning rate scheduler is set to 100 times the number of warmup epochs."
            )
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr
        )

        lr_sched_config = {"scheduler": lr_sched, "interval": "step"}

        return {"optimizer": opt, "lr_scheduler": lr_sched_config}

    @classmethod
    def _load(cls, path, map_location, **submodules):
        d = torch.load(path, map_location)

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")

        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
            if key not in submodules
        }

        if not hasattr(submodules["predictor"].criterion, "_defaults"):
            submodules["predictor"].criterion = submodules["predictor"].criterion.__class__(
                task_weights=submodules["predictor"].criterion.task_weights
            )

        return submodules, state_dict, hparams

    @classmethod
    def _add_metric_task_weights_to_state_dict(cls, state_dict, hparams):
        if "metrics.0.task_weights" not in state_dict:
            metrics = hparams["metrics"]
            n_metrics = len(metrics) if metrics is not None else 1
            for i_metric in range(n_metrics):
                state_dict[f"metrics.{i_metric}.task_weights"] = torch.tensor([[1.0]])
            state_dict[f"metrics.{i_metric + 1}.task_weights"] = state_dict[
                "predictor.criterion.task_weights"
            ]
        return state_dict

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs
    ) -> MPNN:
        submodules = {
            k: v for k, v in kwargs.items() if k in ["message_passing", "agg", "predictor"]
        }
        submodules, state_dict, hparams = cls._load(checkpoint_path, map_location, **submodules)
        kwargs.update(submodules)

        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)
        d = torch.load(checkpoint_path, map_location)
        d["state_dict"] = state_dict
        buffer = io.BytesIO()
        torch.save(d, buffer)
        buffer.seek(0)

        return super().load_from_checkpoint(buffer, map_location, hparams_file, strict, **kwargs)

    @classmethod
    def load_from_file(cls, model_path, map_location=None, strict=True, **submodules) -> MPNN:
        submodules, state_dict, hparams = cls._load(model_path, map_location, **submodules)
        hparams.update(submodules)

        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)

        model = cls(**hparams)
        model.load_state_dict(state_dict, strict=strict)

        return model


class MolAtomBondMPNN(pl.LightningModule):
    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        mol_predictor: Predictor,
        atom_predictor: Predictor,
        bond_predictor: Predictor,
        batch_norm: bool = False,
        metrics: Iterable[ChempropMetric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
    ):
        super().__init__()
        # manually add X_d_transform to hparams to suppress lightning's warning about double saving
        # its state_dict values.
        self.save_hyperparameters(ignore=["X_d_transform", "message_passing", "agg", "predictor"])
        self.hparams["X_d_transform"] = X_d_transform
        self.hparams.update(
            {
                "message_passing": message_passing.hparams,
                "agg": agg.hparams,
                "predictor": predictor.hparams,
            }
        )

        self.message_passing = message_passing
        self.agg = agg
        self.bn = nn.BatchNorm1d(self.message_passing.output_dim) if batch_norm else nn.Identity()
        self.mol_predictor = mol_predictor
        self.atom_predictor = atom_predictor
        self.bond_predictor = bond_predictor

        self.X_d_transform = X_d_transform if X_d_transform is not None else nn.Identity()

        self.metrics = (
            nn.ModuleList([*metrics, self.criterion.clone()])
            if metrics
            else nn.ModuleList([self.predictor._T_default_metric(), self.criterion.clone()])
        )

        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    @property
    def mol_output_dim(self) -> int:
        return self.mol_predictor.output_dim

    @property
    def atom_output_dim(self) -> int:
        return self.atom_predictor.output_dim

    @property
    def bond_output_dim(self) -> int:
        return self.bond_predictor.output_dim

    @property
    def mol_n_tasks(self) -> int:
        return self.mol_predictor.n_tasks

    @property
    def atom_n_tasks(self) -> int:
        return self.atom_predictor.n_tasks

    @property
    def bond_n_tasks(self) -> int:
        return self.bond_predictor.n_tasks

    @property
    def mol_n_targets(self) -> int:
        return self.mol_predictor.n_targets

    @property
    def atom_n_targets(self) -> int:
        return self.atom_predictor.n_targets

    @property
    def bond_n_targets(self) -> int:
        return self.bond__predictor.n_targets

    @property
    def mol_criterion(self) -> ChempropMetric:
        return self.mol_predictor.criterion

    @property
    def atom_criterion(self) -> ChempropMetric:
        return self.atom_predictor.criterion

    @property
    def bond_criterion(self) -> ChempropMetric:
        return self.bond_predictor.criterion

    def fingerprint(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, E_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """the learned fingerprints for the input molecules"""
        H_v, H_b = self.message_passing(bmg, V_d, E_d)
        H_g = self.agg(H_v, bmg.batch)
        H_g = self.bn(H_g)
        H_g = H_g if X_d is None else torch.cat((H, self.X_d_transform(X_d)), 1)

        return H_g, H_v, H_b

    def encoding(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, E_d: Tensor | None = None, X_d: Tensor | None = None, i: int = -1
    ) -> tuple[Tensor]:
        """Calculate the :attr:`i`-th hidden representation"""
        H_g, H_v, H_b = self.fingerprint(bmg, V_d, E_d, X_d)
        return self.mol_predictor.encode(H_g, i), self.atom_predictor.encode(H_v, i), self.bond_predictor.encode(H_b, i)

    def forward(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, E_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> tuple[Tensor]:
        """Generate predictions for the input molecules/reactions"""
        H_g, H_v, H_b = self.fingerprint(bmg, V_d, E_d, X_d)
        return self.mol_predictor(H_g), self.atom_predictor(H_v), self.bond_predictor(H_b)

    def training_step(self, batch: MixedTrainingBatch, batch_idx):
        bmg, V_d, E_d, X_d, mol_targets, atom_targets, bond_targets, mol_weights, atom_weights, bond_weights,
        mol_lt_mask, atom_lt_mask, bond_lt_mask, mol_gt_mask, atom_gt_mask, bond_gt_mask = batch

        #find a way to split targets, weights, masks.

        mol_mask = mol_targets.isfinite()
        atom_mask = atom_targets.isfinite()
        bond_mask = bond_targets.isfinite()
        mol_targets = mol_targets.nan_to_num(nan=0.0)
        atom_targets = atom_targets.nan_to_num(nan=0.0)
        bond_targets = bond_targets.nan_to_num(nan=0.0)

        Z_g, Z_v, Z_b = self.fingerprint(bmg, V_d, E_d, X_d)
        mol_preds = self.mol_predictor.train_step(Z_g)
        mol_l = self.mol_criterion(mol_preds, mol_targets, mol_mask, mol_weights, mol_lt_mask, mol_gt_mask)
        atom_preds = self.atom_predictor.train_step(Z_v)
        atom_l = self.atom_criterion(atom_preds, atom_targets, atom_mask, atom_weights, atom_lt_mask, atom_gt_mask)
        bond_preds = self.bond_predictor.train_step(Z_b)
        bond_l = self.bond_criterion(bond_preds, bond_targets, bond_mask, bond_weights, bond_lt_mask, bond_gt_mask)        

        self.log("mol_train_loss", self.mol_criterion, prog_bar=True, on_epoch=True)
        self.log("atom_train_loss", self.atom_criterion, prog_bar=True, on_epoch=True)
        self.log("bond_train_loss", self.bond_criterion, prog_bar=True, on_epoch=True)

        return mol_l, atom_l, bond_l

    def on_validation_model_eval(self) -> None:
        self.eval()
        self.mol_predictor.output_transform.train()
        self.atom_predictor.output_transform.train()
        self.bond_predictor.output_transform.train()

    def validation_step(self, batch: MixedTrainingBatch, batch_idx: int = 0):
        self._evaluate_batch(batch, "val")

        bmg, V_d, E_d, X_d, mol_targets, atom_targets, bond_targets, mol_weights, atom_weights, bond_weights,
        mol_lt_mask, atom_lt_mask, bond_lt_mask, mol_gt_mask, atom_gt_mask, bond_gt_mask = batch

        mol_mask = mol_targets.isfinite()
        atom_mask = atom_targets.isfinite()
        bond_mask = bond_targets.isfinite()
        mol_targets = mol_targets.nan_to_num(nan=0.0)
        atom_targets = atom_targets.nan_to_num(nan=0.0)
        bond_targets = bond_targets.nan_to_num(nan=0.0)

        Z_g, Z_v, Z_b = self.fingerprint(bmg, V_d, E_d, X_d)
        mol_preds = self.mol_predictor.train_step(Z_g)
        atom_preds = self.atom_predictor.train_step(Z_v)
        bond_preds = self.bond_predictor.train_step(Z_b)

        self.metrics[-1](mol_preds, mol_targets, mol_mask, mol_weights, mol_lt_mask, mol_gt_mask)
        self.log("mol_val_loss", self.metrics[-1], batch_size=len(batch[0]), prog_bar=True)

        self.metrics[-1](atom_preds, atom_targets, atom_mask, atom_weights, atom_lt_mask, atom_gt_mask)
        self.log("atom_val_loss", self.metrics[-1], batch_size=len(batch[0]), prog_bar=True)

        self.metrics[-1](bond_preds, bond_targets, bond_mask, bond_weights, bond_lt_mask, bond_gt_mask)
        self.log("bond_val_loss", self.metrics[-1], batch_size=len(batch[0]), prog_bar=True)

    def test_step(self, batch: MixedTrainingBatch, batch_idx: int = 0):
        self._evaluate_batch(batch, "test")

    def _evaluate_batch(self, batch: MixedTrainingBatch, label: str) -> None:
        bmg, V_d, E_d, X_d, mol_targets, atom_targets, bond_targets, mol_weights, atom_weights, bond_weights,
        mol_lt_mask, atom_lt_mask, bond_lt_mask, mol_gt_mask, atom_gt_mask, bond_gt_mask = batch

        mol_mask = mol_targets.isfinite()
        atom_mask = atom_targets.isfinite()
        bond_mask = bond_targets.isfinite()
        mol_targets = mol_targets.nan_to_num(nan=0.0)
        atom_targets = atom_targets.nan_to_num(nan=0.0)
        bond_targets = bond_targets.nan_to_num(nan=0.0)
        mol_preds, atom_preds, bond_preds = self(bmg, V_d, E_d, X_d)
        mol_weights = torch.ones_like(mol_weights)
        atom_weights = torch.ones_like(atom_weights)
        bond_weights = torch.ones_like(bond_weights)

        if self.predictor.n_targets > 1:
            preds = preds[..., 0]

        for m in self.metrics[:-1]:
            m.update(mol_preds, mol_targets, mol_mask, mol_weights, mol_lt_mask, mol_gt_mask)
            m.update(atom_preds, atom_targets, atom_mask, atom_weights, atom_lt_mask, atom_gt_mask)
            m.update(bond_preds, bond_targets, bond_mask, bond_weights, bond_lt_mask, bond_gt_mask)            
            self.log(f"{label}/{m.alias}", m, batch_size=len(batch[0]))

    def predict_step(self, batch: MixedTrainingBatch, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Return the predictions of the input batch

        Parameters
        ----------
        batch : MixedTrainingBatch
            the input batch

        Returns
        -------
        Tensor
            a tensor of varying shape depending on the task type:

            * regression/binary classification: ``n x (t * s)``, where ``n`` is the number of input
              molecules/reactions, ``t`` is the number of tasks, and ``s`` is the number of targets
              per task. The final dimension is flattened, so that the targets for each task are
              grouped. I.e., the first ``t`` elements are the first target for each task, the second
              ``t`` elements the second target, etc.

            * multiclass classification: ``n x t x c``, where ``c`` is the number of classes
        """
        bmg, X_vd, X_ed, X_d, *_ = batch

        return self(bmg, X_vd, X_ed, X_d)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)
        if self.trainer.train_dataloader is None:
            # Loading `train_dataloader` to estimate number of training batches.
            # Using this line of code can pypass the issue of using `num_training_batches` as described [here](https://github.com/Lightning-AI/pytorch-lightning/issues/16060).
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            warnings.warn(
                "For infinite training, the number of cooldown epochs in learning rate scheduler is set to 100 times the number of warmup epochs."
            )
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr
        )

        lr_sched_config = {"scheduler": lr_sched, "interval": "step"}

        return {"optimizer": opt, "lr_scheduler": lr_sched_config}

    @classmethod
    def _load(cls, path, map_location, **submodules):
        d = torch.load(path, map_location)

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")

        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
            if key not in submodules
        }

        if not hasattr(submodules["predictor"].criterion, "_defaults"):
            submodules["predictor"].criterion = submodules["predictor"].criterion.__class__(
                task_weights=submodules["predictor"].criterion.task_weights
            )

        return submodules, state_dict, hparams

    @classmethod
    def _add_metric_task_weights_to_state_dict(cls, state_dict, hparams):
        if "metrics.0.task_weights" not in state_dict:
            metrics = hparams["metrics"]
            n_metrics = len(metrics) if metrics is not None else 1
            for i_metric in range(n_metrics):
                state_dict[f"metrics.{i_metric}.task_weights"] = torch.tensor([[1.0]])
            state_dict[f"metrics.{i_metric + 1}.task_weights"] = state_dict[
                "predictor.criterion.task_weights"
            ]
        return state_dict

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs
    ) -> MPNN:
        submodules = {
            k: v for k, v in kwargs.items() if k in ["message_passing", "agg", "predictor"]
        }
        submodules, state_dict, hparams = cls._load(checkpoint_path, map_location, **submodules)
        kwargs.update(submodules)

        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)
        d = torch.load(checkpoint_path, map_location)
        d["state_dict"] = state_dict
        buffer = io.BytesIO()
        torch.save(d, buffer)
        buffer.seek(0)

        return super().load_from_checkpoint(buffer, map_location, hparams_file, strict, **kwargs)

    @classmethod
    def load_from_file(cls, model_path, map_location=None, strict=True, **submodules) -> MPNN:
        submodules, state_dict, hparams = cls._load(model_path, map_location, **submodules)
        hparams.update(submodules)

        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)

        model = cls(**hparams)
        model.load_state_dict(state_dict, strict=strict)

        return model
