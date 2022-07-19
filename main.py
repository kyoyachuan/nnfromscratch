from nnfromscratch import (
    TwoLayerNetwork,
    SGD, Momentum, Adam,
    BinaryCrossEntropy, MeanSquareError
)
from utils.data import generate_linear, generate_XOR_easy
from utils.metrics import accuracy
from utils.plot import plot_result
from utils.exp_manager import ExperimentManager, ExperimentCfg


@ExperimentManager('config/exp.toml')
def main(cfg: ExperimentCfg) -> dict:
    """
    Main function.

    Args:
        cfg (ExperimentCfg): experiment configuration

    Returns:
        dict: results of the experiment group
    """
    # Generate data
    if cfg.trainer['dataset'] == 'linear':
        x_train, y_train = generate_linear()
        x_test, y_test = generate_linear()
    elif cfg.trainer['dataset'] == 'xor_easy':
        x_train, y_train = generate_XOR_easy()
        x_test, y_test = generate_XOR_easy()

    # Get loss
    if cfg.trainer['loss'] == 'binary_cross_entropy':
        loss = BinaryCrossEntropy()
    elif cfg.trainer['loss'] == 'mean_square_error':
        loss = MeanSquareError()

    # Create model
    model = TwoLayerNetwork(
        input_size=2,
        hidden_size_1=cfg.trainer['hidden_size'],
        hidden_size_2=cfg.trainer['hidden_size'],
        output_size=1,
        activation=cfg.trainer['activation'],
        loss=loss
    )
    print(model)

    # Train model
    if cfg.trainer['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=cfg.trainer['lr'])
    elif cfg.trainer['optimizer'] == 'momentum':
        optimizer = Momentum(model.parameters(), lr=cfg.trainer['lr'])
    elif cfg.trainer['optimizer'] == 'sgd':
        optimizer = SGD(model.parameters(), lr=cfg.trainer['lr'])

    loss = []
    acc = []
    for i in range(cfg.epochs):
        y_hat = model(x_train)
        loss.append(model.loss(y_hat, t=y_train))
        acc.append(accuracy(y_hat, y_train))

        model.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 200 == 0:
            print(f'epoch: {i} loss: {loss[-1]} acc: {acc[-1]}')

    # Plot result
    y_pred = model(x_test)
    print(y_pred)
    plot_result(f'result/pred_{cfg.name}_{cfg.exp_value}.png', x_test, y_test, y_pred > 0.5)

    return {
        'loss': loss,
        'accuracy': acc
    }


if __name__ == '__main__':
    main()
