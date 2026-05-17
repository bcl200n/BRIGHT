"""General utility functions for training and evaluation (PaddlePaddle version)."""

import collections
import datetime
import logging
import random
import sys
import time

import numpy as np
import paddle
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def collate_fn(batch: list) -> tuple:
    return tuple(zip(*batch))


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


class SmoothedValue:
    """Track a series of values and provide access to smoothed values."""

    def __init__(self, window_size: int = 20, fmt: str = "{median:.4f} ({global_avg:.4f})"):
        self.deque: collections.deque = collections.deque(maxlen=window_size)
        self.total: float = 0.0
        self.count: int = 0
        self.fmt: str = fmt

    def update(self, value: float, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self) -> float:
        if not self.deque:
            return 0.0
        d = np.array(list(self.deque))
        return float(np.median(d))

    @property
    def avg(self) -> float:
        if not self.deque:
            return 0.0
        d = np.array(list(self.deque), dtype=np.float32)
        return float(d.mean())

    @property
    def global_avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def max(self) -> float:
        if not self.deque:
            return 0.0
        return max(self.deque)

    @property
    def value(self) -> float:
        if not self.deque:
            return 0.0
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    _logger_id = 0

    def __init__(self, delimiter: str = "\t", log_file: str = None):
        self.meters: dict[str, SmoothedValue] = collections.defaultdict(SmoothedValue)
        self.delimiter: str = delimiter

        MetricLogger._logger_id += 1
        self.logger = logging.getLogger(f"metric_logger.{MetricLogger._logger_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        if log_file:
            self.logger.addHandler(logging.FileHandler(log_file, mode="a"))

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, paddle.Tensor):
                v = float(v.numpy())
            assert isinstance(v, (float, int)), f"Expected float or int, got {type(v)}"
            self.meters[k].update(v)

    def __getattr__(self, attr: str):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")

    def __str__(self) -> str:
        entries = []
        for name, meter in self.meters.items():
            entries.append(f"{name}: {str(meter)}")
        return self.delimiter.join(entries)

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def log_every(self, iterable, print_freq: int, header: str = ""):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = self.delimiter.join([
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ])

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                self.logger.info(
                    log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                    )
                )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        n_iter = max(len(iterable), 1)
        self.logger.info(f"{header} Total time: {total_time_str} ({total_time / n_iter:.4f} s / it)")
