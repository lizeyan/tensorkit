from typing import Any, Optional, List, Tuple, Callable

import numpy as np

import mltk
import tensorkit as tk
from tensorkit import tensor as T


def print_experiment_summary(exp: mltk.Experiment,
                             train_data: Any,  # anything that has '__len__'
                             val_data: Optional[Any] = None,
                             test_data: Optional[Any] = None,
                             printer: Optional[Callable[[str], Any]] = print):
    # the config
    mltk.print_config(exp.config)
    printer('')

    # the dataset info
    data_info = []
    for name, data in [('Train', train_data), ('Validation', val_data),
                         ('Test', test_data)]:
        if data is not None:
            data_info.append((name, len(data)))
    if data_info:
        printer(mltk.format_key_values(data_info, 'Number of Data'))
        printer('')

    # the device info
    device_info = [
        ('Current', T.current_device())
    ]
    gpu_devices = T.gpu_device_list()
    if gpu_devices:
        device_info.append(('Available', gpu_devices))
    printer(mltk.format_key_values(device_info, 'Device Info'))
    printer('')


def print_parameters_summary(
        params: List[T.Variable], names: List[str], printer: Optional[Callable[[str], Any]] = print
):
    shapes = []
    sizes = []
    total_size = 0
    max_shape_len = 0
    max_size_len = 0
    right_pad = ' ' * 3

    for param in params:
        shape = T.shape(param)
        size = np.prod(shape)
        total_size += size
        shapes.append(str(shape))
        sizes.append(f'{size:,d}')
        max_shape_len = max(max_shape_len, len(shapes[-1]))
        max_size_len = max(max_size_len, len(sizes[-1]))

    total_size = f'{total_size:,d}'
    right_len = max(max_shape_len + len(right_pad) + max_size_len, len(total_size))

    param_info = []
    max_name_len = 0
    for param, name, shape, size in zip(params, names, shapes, sizes):
        max_name_len = max(max_name_len, len(name))
        right = f'{shape:<{max_shape_len}s}{right_pad}{size:>{max_size_len}s}'
        right = f'{right:>{right_len}s}'
        param_info.append((name, right))

    if param_info:
        param_info.append(('Total', f'{total_size:>{right_len}s}'))
        lines = mltk.format_key_values(
            param_info, title='Parameters', formatter=str).strip().split('\n')
        k = len(lines[-1])
        lines.insert(-1, '-' * k)

        printer('\n'.join(lines))


def get_weights_and_names(layer: T.Module) -> Tuple[List[T.Variable], List[str]]:
    params = []
    names = []
    for name, param in tk.layers.iter_named_parameters(layer):
        if not name.endswith('.bias_store.value') and not name.endswith('.bias'):
            params.append(param)
            names.append(name)
    return params, names


def get_params_and_names(layer: T.Module
                         ) -> Tuple[List[T.Variable], List[str]]:
    params = []
    names = []
    for name, param in tk.layers.iter_named_parameters(layer):
        params.append(param)
        names.append(name)
    return params, names