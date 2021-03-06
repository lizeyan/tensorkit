from functools import partial
from typing import *

import mltk
import tensorkit as tk
import tensorkit.utils.misc
from tensorkit import tensor as T
from tensorkit.examples import utils
from tensorkit.typing_ import TensorOrData


class Config(mltk.Config):
    # model parameters
    z_dim: int = 40
    flow_levels: int = 20
    coupling_hidden_layer_count: int = 1
    coupling_hidden_layer_units: int = 250
    coupling_layer_scale: str = 'sigmoid'
    strict_invertible: bool = False
    qz_logstd_min: Optional[float] = -7
    l2_reg: float = 0.0001

    # initialization parameters
    init_batch_count: int = 10

    # train parameters
    max_epoch: int = 2400
    batch_size: int = 128
    warmup_epochs: Optional[int] = 100
    initial_lr: float = 0.001
    lr_anneal_ratio: float = 0.1
    lr_anneal_epochs: int = 800
    global_clip_norm: Optional[float] = 100.0

    # evaluation parameters
    test_n_z: int = 500
    test_batch_size: int = 256


class VAE(tk.layers.BaseLayer):

    x_dim: int
    config: Config

    def __init__(self, x_dim: int, config: Config):
        super().__init__()
        self.x_dim = x_dim
        self.config = config

        # common nn parameters
        layer_args = tk.layers.LayerArgs(). \
            set_args(['dense'], activation=tk.layers.LeakyReLU)

        # nn for q(z|x)
        q_builder = tk.layers.SequentialBuilder(x_dim, layer_args=layer_args)
        self.hx_for_qz = q_builder.dense(500).dense(500).build()
        self.qz_mean = q_builder.as_input().linear(config.z_dim).build()
        self.qz_logstd = q_builder.as_input().linear(config.z_dim).build()

        # the posterior flow
        flows = []
        for i in range(config.flow_levels):
            # act norm
            flows.append(tk.flows.ActNorm(config.z_dim))

            # coupling layer
            n1 = config.z_dim // 2
            n2 = config.z_dim - n1
            b = tk.layers.SequentialBuilder(n1, layer_args=layer_args)
            for j in range(config.coupling_hidden_layer_count):
                b.dense(config.coupling_hidden_layer_units)
            shift_and_pre_scale = tk.layers.Branch(
                branches=[
                    # shift
                    b.as_input().linear(n2, weight_init=tk.init.zeros).build(),
                    # pre_scale
                    b.as_input().linear(n2, weight_init=tk.init.zeros).build(),
                ],
                shared=b.build(),
            )
            flows.append(tk.flows.CouplingLayer(
                shift_and_pre_scale, scale=config.coupling_layer_scale))

            # feature rearrangement by invertible dense
            flows.append(tk.flows.InvertibleDense(
                config.z_dim, strict=config.strict_invertible))
        self.posterior_flow = tk.flows.SequentialFlow(flows)

        # nn for p(x|z)
        p_builder = tk.layers.SequentialBuilder(config.z_dim, layer_args=layer_args)
        self.px_logits = p_builder.dense(500).dense(500).linear(x_dim).build(True)

    def initialize(self, x):
        x = T.as_tensor(x)
        _ = self.get_chain(x).vi.training.sgvb()  # trigger initialization
        tk.layers.jit_compile_children(self)
        _ = self.get_chain(x).vi.training.sgvb()  # trigger JIT

    def q(self,
          x: T.Tensor,
          observed: Optional[Mapping[str, TensorOrData]] = None,
          n_z: Optional[int] = None) -> tk.BayesianNet:
        net = tk.BayesianNet(observed=observed)
        hx = self.hx_for_qz(T.cast(x, dtype=T.float32))
        z_mean = self.qz_mean(hx)
        z_logstd = self.qz_logstd(hx)
        z_logstd = T.maybe_clip(z_logstd, min_val=self.config.qz_logstd_min)
        qz = tk.FlowDistribution(
            tk.Normal(mean=z_mean, logstd=z_logstd, event_ndims=1),
            self.posterior_flow,
        )
        z = net.add('z', qz, n_samples=n_z)
        return net

    def p(self,
          observed: Optional[Mapping[str, TensorOrData]] = None,
          n_z: Optional[int] = None) -> tk.BayesianNet:
        net = tk.BayesianNet(observed=observed)

        # sample z ~ p(z)
        z = net.add('z', tk.UnitNormal([1, self.config.z_dim], event_ndims=1),
                    n_samples=n_z)
        x_logits = self.px_logits(z.tensor)
        x = net.add('x', tk.Bernoulli(logits=x_logits, event_ndims=1))
        return net

    def get_chain(self, x, n_z: Optional[int] = None):
        latent_axis = 0 if n_z is not None else None
        return self.q(x, n_z=n_z).chain(
            self.p, observed={'x': x}, n_z=n_z, latent_axis=latent_axis)


def main(exp: mltk.Experiment[Config]):
    # prepare the data
    train_stream, _, test_stream = utils.get_mnist_streams(
        batch_size=exp.config.batch_size,
        test_batch_size=exp.config.test_batch_size,
        flatten=True,
        x_range=(0., 1.),
        use_y=False,
        mapper=utils.BernoulliSampler().as_mapper(),
    )

    tensorkit.utils.misc.print_experiment_summary(
        exp, train_data=train_stream, test_data=test_stream)

    # build the network
    vae: VAE = VAE(train_stream.data_shapes[0][0], exp.config)
    params, param_names = tensorkit.utils.misc.get_params_and_names(vae)
    tensorkit.utils.misc.print_parameters_summary(params, param_names)
    print('')
    mltk.print_with_time('Network constructed.')

    # initialize the network with first few batches of train data
    [init_x] = train_stream.get_arrays(max_batch=exp.config.init_batch_count)
    vae.initialize(init_x)
    mltk.print_with_time('Network initialized')

    # define the train and evaluate functions
    def train_step(x):
        chain = vae.get_chain(x)

        # loss with beta
        beta = min(loop.epoch / 100., 1.)
        log_qz_given_x = T.reduce_mean(chain.q['z'].log_prob())
        log_pz = T.reduce_mean(chain.p['z'].log_prob())
        log_px_given_z = T.reduce_mean(chain.p['x'].log_prob())
        kl = log_pz - log_qz_given_x
        elbo = log_px_given_z + beta * kl

        # add regularization
        loss = -elbo + exp.config.l2_reg * T.nn.l2_regularization(params)

        # construct the train metrics
        ret = {'loss': loss, 'kl': kl, 'log p(x|z)': log_px_given_z,
               'log q(z|x)': log_qz_given_x, 'log p(z)': log_pz}
        if loop.epoch >= 100:
            ret['elbo'] = elbo
        return ret

    def eval_step(x, n_z=exp.config.test_n_z):
        with tk.layers.scoped_eval_mode(vae), T.no_grad():
            chain = vae.get_chain(x, n_z=n_z)
            log_qz_given_x = T.reduce_mean(chain.q['z'].log_prob())
            log_pz = T.reduce_mean(chain.p['z'].log_prob())
            log_px_given_z = T.reduce_mean(chain.p['x'].log_prob())
            kl = log_pz - log_qz_given_x
            elbo = log_px_given_z + kl
            nll = -chain.vi.evaluation.is_loglikelihood(reduction='mean')
        return {'elbo': elbo, 'nll': nll, 'kl': kl, 'log p(x|z)': log_px_given_z,
               'log q(z|x)': log_qz_given_x, 'log p(z)': log_pz}

    def plot_samples(epoch=None):
        epoch = epoch or loop.epoch
        with tk.layers.scoped_eval_mode(vae), T.no_grad():
            logits = vae.p(n_z=100)['x'].distribution.logits
            images = T.reshape(
                T.cast(T.clip(T.nn.sigmoid(logits) * 255., 0., 255.), dtype=T.uint8),
                [-1, 28, 28],
            )
        utils.save_images_collection(
            images=T.to_numpy(images),
            filename=exp.abspath(f'plotting/{epoch}.png'),
            grid_size=(10, 10),
        )

    # build the optimizer and the train loop
    loop = mltk.TrainLoop(max_epoch=exp.config.max_epoch)
    loop.add_callback(mltk.callbacks.StopOnNaN())
    optimizer = tk.optim.Adam(params)
    lr_scheduler = tk.optim.lr_scheduler.AnnealingLR(
        optimizer=optimizer,
        initial_lr=exp.config.initial_lr,
        ratio=exp.config.lr_anneal_ratio,
        epochs=exp.config.lr_anneal_epochs
    )
    lr_scheduler.bind(loop)
    loop.run_after_every(
        lambda: loop.test().run(partial(eval_step, n_z=10), test_stream),
        epochs=10
    )
    loop.run_after_every(plot_samples, epochs=10)

    # train the model
    tk.layers.set_train_mode(vae, True)
    utils.fit_model(
        loop=loop, optimizer=optimizer, fn=train_step, stream=train_stream,
        global_clip_norm=exp.config.global_clip_norm, param_names=param_names,
    )

    # do the final test
    results = mltk.TestLoop().run(eval_step, test_stream)
    plot_samples('final')


if __name__ == '__main__':
    with mltk.Experiment(Config) as exp:
        with T.use_device(T.first_gpu_device()):
            main(exp)
