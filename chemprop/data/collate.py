from dataclasses import InitVar, dataclass, field
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import torch
from torch import Tensor

from chemprop.data.datasets import MixedDatum, Datum
from chemprop.data.molgraph import MolGraph


@dataclass(repr=False, eq=False, slots=True)
class BatchMolGraph:
    """A :class:`BatchMolGraph` represents a batch of individual :class:`MolGraph`\s.

    It has all the attributes of a ``MolGraph`` with the addition of the ``batch`` attribute. This
    class is intended for use with data loading, so it uses :obj:`~torch.Tensor`\s to store data
    """

    mgs: InitVar[Sequence[MolGraph]]
    """A list of individual :class:`MolGraph`\s to be batched together"""
    V: Tensor = field(init=False)
    """the atom feature matrix"""
    E: Tensor = field(init=False)
    """the bond feature matrix"""
    edge_index: Tensor = field(init=False)
    """an tensor of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: Tensor = field(init=False)
    """A tensor of shape ``E`` that maps from an edge index to the index of the source of the
    reverse edge in the ``edge_index`` attribute."""
    batch: Tensor = field(init=False)
    """the index of the parent :class:`MolGraph` in the batched graph"""

    __size: int = field(init=False)

    def __post_init__(self, mgs: Sequence[MolGraph]):
        self.__size = len(mgs)

        Vs = []
        Es = []
        edge_indexes = []
        rev_edge_indexes = []
        batch_indexes = []

        num_nodes = 0
        num_edges = 0
        for i, mg in enumerate(mgs):
            Vs.append(mg.V)
            Es.append(mg.E)
            edge_indexes.append(mg.edge_index + num_nodes)
            rev_edge_indexes.append(mg.rev_edge_index + num_edges)
            batch_indexes.append([i] * len(mg.V))

            num_nodes += mg.V.shape[0]
            num_edges += mg.edge_index.shape[1]

        self.V = torch.from_numpy(np.concatenate(Vs)).float()
        self.E = torch.from_numpy(np.concatenate(Es)).float()
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes)).long()
        self.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes)).long()
        self.batch = torch.tensor(np.concatenate(batch_indexes)).long()

    def __len__(self) -> int:
        """the number of individual :class:`MolGraph`\s in this batch"""
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)


class TrainingBatch(NamedTuple):
    bmg: BatchMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_batch(batch: Iterable[Datum]) -> TrainingBatch:
    mgs, V_ds, E_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    # if ys[0] is not None:
    #     dim = ys[0].shape[1]
    #     np_y = np.empty((0, dim), float)
    #     for y in ys:
    #         np_y = np.vstack([np_y, y])
    # if lt_masks[0] is not None:
    #     dim = lt_masks[0].shape[1]
    #     np_lt = np.empty((0, dim))
    #     for lt in lt_masks:
    #         np_lt = np.vstack([np_lt, lt])
    # if gt_masks[0] is not None:
    #     dim = gt_masks[0].shape[1]
    #     np_gt = np.empty((0, dim))
    #     for gt in gt_masks:
    #         np_gt = np.vstack([np_gt, gt])

    np_y = np.vstack(ys)
    np_lt = np.vstack(lt_masks)
    np_gt = np.vstack(gt_masks)

    if np_y.shape[0] == len(ys):
        weights_tensor = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
    else:
        num_atoms = torch.tensor([y.shape[0] for y in ys])
        weights_tensor = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
        weights_tensor = torch.repeat_interleave(weights_tensor, repeats=num_atoms)

    return TrainingBatch(
        BatchMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np_y).float(),
        weights_tensor,
        None if lt_masks[0] is None else torch.from_numpy(np_lt),
        None if gt_masks[0] is None else torch.from_numpy(np_gt),
    )


class MixedTrainingBatch(NamedTuple):
    bmg: BatchMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    mol_Y: Tensor | None
    atom_Y: Tensor | None
    bond_Y: Tensor | None
    mol_w: Tensor
    atom_w: Tensor
    bond_w: Tensor
    mol_lt_mask: Tensor | None
    atom_lt_mask: Tensor | None
    bond_lt_mask: Tensor | None
    mol_gt_mask: Tensor | None
    atom_gt_mask: Tensor | None
    bond_gt_mask: Tensor | None


def mixed_collate_batch(batch: Iterable[MixedDatum]) -> MixedTrainingBatch:
    mgs, V_ds, E_ds, x_ds, ys, weights, lt_masks, gt_masks, flags = zip(*batch)

    mol_indices, atom_indices, bond_indices = [], [], []
    for y in range(len(ys)):
        if flags[y] == "mol":
            mol_indices.append(y)
        elif flags[y] == "atom":
            atom_indices.append(y)
        else:
            bond_indices.append(y)

    mol_np_y = None if mol_indices[0] is None else torch.from_numpy(np.vstack([ys[i] for i in mol_indices])).float()
    atom_np_y = None if len(atom_indices) == 0 else torch.from_numpy(np.vstack([ys[i] for i in atom_indices])).float()
    bond_np_y = None if len(bond_indices) == 0 else torch.from_numpy(np.vstack([ys[i] for i in bond_indices])).float()
    for i in mol_indices:
        print(lt_masks[i])
    mol_np_lt = None if len(mol_indices) == 0  else torch.from_numpy(np.vstack([lt_masks[i] for i in mol_indices])).float()
    atom_np_lt = None if len(atom_indices) == 0 else torch.from_numpy(np.vstack([lt_masks[i] for i in atom_indices])).float()
    bond_np_lt = None if len(bond_indices) == 0 else torch.from_numpy(np.vstack([lt_masks[i] for i in bond_indices])).float()
    mol_np_gt = None if len(mol_indices) == 0  else torch.from_numpy(np.vstack([gt_masks[i] for i in mol_indices])).float()
    atom_np_gt = None if len(atom_indices) == 0 else torch.from_numpy(np.vstack([gt_masks[i] for i in atom_indices])).float()
    bond_np_gt = None if len(bond_indices) == 0 else torch.from_numpy(np.vstack([gt_masks[i] for i in bond_indices])).float()

    mol_weights_tensor = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
    num_atoms = torch.tensor([ys[j].shape[0] for j in atom_indices])
    atom_weights_tensor = torch.repeat_interleave(mol_weights_tensor, repeats=num_atoms)
    num_bonds = torch.tensor([ys[j].shape[0] for j in bond_indices])
    bond_weights_tensor = torch.repeat_interleave(mol_weights_tensor, repeats=num_bonds)

    return MixedTrainingBatch(
        BatchMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if E_ds[0] is None else torch.from_numpy(np.concatenate(E_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        mol_np_y,
        atom_np_y,
        bond_np_y,
        mol_weights_tensor,
        atom_weights_tensor,
        bond_weights_tensor,
        mol_np_lt,
        atom_np_lt,
        bond_np_lt,
        mol_np_gt,
        atom_np_gt,
        bond_np_gt,
    )

def mixed_multi_collate_batch(batches: Iterable[Iterable[MixedDatum]]) -> MixedMultiTrainingBatch:
    tbs = [mixed_collate_batch(batch) for batch in zip(*batches)]
    return MixedMultiTrainingBatch (
        *[ [getattr(tb, attr) for tb in tbs] for attr in ('bmg', 'V_d', 'X_d', 'mol_Y', 'atom_Y', 'bond_Y', 'mol_w', 'atom_w', 
            'bond_w', 'mol_lt_mask', 'atom_lt_mask', 'bond_lt_mask', 'mol_gt_mask', 'atom_gt_mask', 'bond_gt_mask') ]
    )

class MulticomponentTrainingBatch(NamedTuple):
    bmgs: list[BatchMolGraph]
    V_ds: list[Tensor | None]
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_multicomponent(batches: Iterable[Iterable[Datum]]) -> MulticomponentTrainingBatch:
    tbs = [collate_batch(batch) for batch in zip(*batches)]

    return MulticomponentTrainingBatch(
        [tb.bmg for tb in tbs],
        [tb.V_d for tb in tbs],
        tbs[0].X_d,
        tbs[0].Y,
        tbs[0].w,
        tbs[0].lt_mask,
        tbs[0].gt_mask,
    )
