from __future__ import annotations

import json
import os
from typing import Iterator

import numpy as np
import numpy.typing as npt


class StreamingDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        context_length: int,
        max_samples_per_iter: int,
        seed: int = 42,
    ) -> None:
        self.batch_size = batch_size
        self.context_length = context_length
        self.max_samples_per_iter = max_samples_per_iter
        self.seed = seed

        self.meta: dict | None = None

        self._data: npt.NDArray[np.uint16] | None = None
        self._n_pos: int | None = None
        self._num_samples: int | None = None
        self._epoch = 0

        self.load(data_dir)

    # ---------------- 加载 ----------------

    def load(self, out_dir: str) -> None:
        meta_path = os.path.join(out_dir, "meta.json")
        data_path = os.path.join(out_dir, "tokens.bin")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.meta = meta

        dtype = np.dtype(meta["dtype"])
        total_len = meta["total_len"]
        actual = os.path.getsize(data_path)
        if actual != total_len * dtype.itemsize:
            raise ValueError(
                f"tokens.bin 大小 {actual} != 期望 {total_len * dtype.itemsize}, 文件不完整，请重新生成")

        # 所有可能的样本起始位置: arange(total_len - (context_length + 1) + 1)
        n_pos = total_len - self.context_length
        if n_pos <= 0:
            raise ValueError(
                f"数据集长度 {total_len} 不足 context_length={self.context_length}")
        elif n_pos < self.batch_size:
            raise ValueError(
                f"可采样位置 {n_pos} 不足一个 batch={self.batch_size}")
        self._n_pos = n_pos
        num_samples = min(self.max_samples_per_iter, n_pos)
        num_samples -= num_samples % self.batch_size
        self._num_samples = num_samples

        if self._data is not None:
            self.close()
        self._data = np.memmap(data_path, dtype=dtype, mode="r", shape=(total_len,))

    def close(self) -> None:
        if self._data is not None:
            self._data._mmap.close()
            self._data = None

    # ---------------- 属性 ----------------

    @property
    def num_tokens(self) -> int:
        return int(self.meta["total_len"])

    @property
    def num_docs(self) -> int:
        return int(self.meta["num_docs"])

    @property
    def num_batches_per_epoch(self) -> int:
        return self._num_samples // self.batch_size

    @property
    def epoch(self) -> int:
        return self._epoch

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self) -> int:
        return self.num_batches_per_epoch

    # ---------------- 迭代 ----------------

    def __iter__(
        self,
    ) -> Iterator[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        data = self._data
        bs = self.batch_size
        ctx = self.context_length
        row = np.arange(ctx + 1)
        n_pos = self._n_pos
        m = self._num_samples

        rng = np.random.default_rng(self.seed + self._epoch)
        starts = rng.choice(n_pos, size=m, replace=False)
        self._epoch += 1

        for i in range(0, m, bs):
            idx = starts[i:i + bs, None] + row
            batch = data[idx]
            yield batch[:, :-1].astype(np.int64), batch[:, 1:].astype(np.int64)
