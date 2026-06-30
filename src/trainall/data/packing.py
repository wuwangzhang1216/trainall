"""Sequence packing for efficient pretraining / CPT.

Short tokenised examples waste compute when padded to a fixed length.  Packing
greedily concatenates token-id lists into fixed-length bins (first-fit-style
greedy bin packing) so almost every position carries a real token.  A trailing
partial bin is padded with ``pad_id``.
"""
from __future__ import annotations

from typing import List, Sequence

__all__ = ["pack_sequences", "Packer"]


def pack_sequences(
    sequences: Sequence[Sequence[int]],
    max_len: int,
    pad_id: int = 0,
    drop_last: bool = False,
) -> List[List[int]]:
    """Greedily pack token-id lists into bins of length ``max_len``.

    Sequences longer than ``max_len`` are chunked.  Each output bin is exactly
    ``max_len`` long; the final bin is padded with ``pad_id`` unless empty (or
    dropped when ``drop_last``).
    """
    if max_len <= 0:
        raise ValueError("max_len must be positive")

    packed: List[List[int]] = []
    cur: List[int] = []
    for seq in sequences:
        seq = list(seq)
        i = 0
        while i < len(seq):
            room = max_len - len(cur)
            cur.extend(seq[i : i + room])
            i += room
            if len(cur) == max_len:
                packed.append(cur)
                cur = []

    if cur:
        if not drop_last:
            cur.extend([pad_id] * (max_len - len(cur)))
            packed.append(cur)
    return packed


class Packer:
    """Stateful streaming wrapper around :func:`pack_sequences`.

    Feed token-id lists incrementally with :meth:`add`; full bins are emitted as
    they fill.  Call :meth:`flush` at the end to drain (and pad) the remainder.
    """

    def __init__(self, max_len: int, pad_id: int = 0) -> None:
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        self.max_len = max_len
        self.pad_id = pad_id
        self._buf: List[int] = []

    def add(self, sequence: Sequence[int]) -> List[List[int]]:
        """Add one tokenised sequence; return any newly completed bins."""
        out: List[List[int]] = []
        seq = list(sequence)
        i = 0
        while i < len(seq):
            room = self.max_len - len(self._buf)
            self._buf.extend(seq[i : i + room])
            i += room
            if len(self._buf) == self.max_len:
                out.append(self._buf)
                self._buf = []
        return out

    def flush(self, pad: bool = True) -> List[List[int]]:
        """Emit and clear the partial buffer (padded to ``max_len`` if ``pad``)."""
        if not self._buf:
            return []
        rem = self._buf
        self._buf = []
        if pad and len(rem) < self.max_len:
            rem = rem + [self.pad_id] * (self.max_len - len(rem))
        return [rem]
