import mltk
import tensorkit as tk
import tensorkit.utils.misc
from tensorkit import tensor as T
from tensorkit.examples import utils


class Config(mltk.Config):
    # initialization parameters
    init_batch_count: int = 32

    # train parameters
    max_epoch: int = 300
    batch_size: int = 64
    initial_lr: float = 0.001
    lr_anneal_ratio: float = 0.1
    lr_anneal_epochs: int = 100

    # test parameters
    test_batch_size: int = 256


def main(exp: mltk.Experiment[Config]):
    # prepare the data
    train_stream, val_stream, test_stream = utils.get_mnist_streams(
        batch_size=exp.config.batch_size,
        test_batch_size=exp.config.test_batch_size,
        val_batch_size=exp.config.test_batch_size,
        val_portion=0.2,
        flatten=True,
        x_range=(-1., 1.),
    )

    tensorkit.utils.misc.print_experiment_summary(
        exp, train_data=train_stream, val_data=val_stream,
        test_data=test_stream
    )

    # build the network
    net: T.Module = tk.layers.SequentialBuilder(784). \
        set_args('dense',
                 activation=tk.layers.LeakyReLU,
                 data_init=tk.init.StdDataInit()). \
        dense(500). \
        dense(500). \
        linear(10). \
        log_softmax(). \
        build()
    params, param_names = tensorkit.utils.misc.get_params_and_names(net)
    tensorkit.utils.misc.print_parameters_summary(params, param_names)
    print('')
    mltk.print_with_time('Network constructed.')

    # initialize the network
    init_x, _ = train_stream.get_arrays(max_batch=exp.config.init_batch_count)
    init_x = T.as_tensor(init_x)
    _ = net(init_x)  # trigger initialization
    net = tk.layers.jit_compile(net)
    _ = net(init_x)  # trigger JIT
    mltk.print_with_time('Network initialized')

    # define the train and evaluate functions
    def train_step(x, y):
        logits = net(x)
        loss = T.nn.cross_entropy_with_logits(logits, y, reduction='mean')
        return {'loss': loss}

    def eval_step(x, y):
        with tk.layers.scoped_eval_mode(net), T.no_grad():
            logits = net(x)
            acc = utils.calculate_acc(logits, y)
        return {'acc': acc}

    # build the optimizer and the train loop
    loop = mltk.TrainLoop(max_epoch=exp.config.max_epoch)
    optimizer = tk.optim.Adam(tk.layers.iter_parameters(net))
    lr_scheduler = tk.optim.lr_scheduler.AnnealingLR(
        optimizer=optimizer,
        initial_lr=exp.config.initial_lr,
        ratio=exp.config.lr_anneal_ratio,
        epochs=exp.config.lr_anneal_epochs
    )
    lr_scheduler.bind(loop)

    # add a callback to do early-stopping on the network parameters
    # according to the validation metric.
    loop.add_callback(
        mltk.callbacks.EarlyStopping(
            checkpoint=tk.train.Checkpoint(net=net),
            root_dir=exp.abspath('./checkpoint/early-stopping'),
            # note for `loop.validation()`, the prefix "val_" will be
            # automatically prepended to any metrics generated by the
            # `evaluate` function.
            metric_name='val_acc',
            smaller_is_better=False,
        )
    )

    # run validation after every 10 epochs
    if val_stream is not None:
        loop.run_after_every(
            lambda: loop.validation().run(eval_step, val_stream),
            epochs=10,
        )

    # run test after every 10 epochs
    loop.run_after_every(
        lambda: loop.test().run(eval_step, test_stream),
        epochs=10,
    )

    # train the model
    tk.layers.set_train_mode(net, True)
    utils.fit_model(loop=loop, optimizer=optimizer, fn=train_step,
                    stream=train_stream)

    # do the final test with the best network parameters (according to validation)
    results = mltk.TestLoop().run(eval_step, test_stream)


if __name__ == '__main__':
    with mltk.Experiment(Config) as exp:
        with T.use_device(T.first_gpu_device()):
            main(exp)
