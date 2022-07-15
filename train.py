from nnfromscratch import TwoLayerNetwork, SGD, BinaryCrossEntropy, MeanSquareError
from utils.data import generate_linear, generate_XOR_easy
from utils.metrics import accuracy
from utils.plot import show_result, show_loss


def main():
    x_train, y_train = generate_linear()
    x_test, y_test = generate_linear()
    # x_train, y_train = generate_XOR_easy()
    # x_test, y_test = generate_XOR_easy()

    model = TwoLayerNetwork(
	    input_size=2,
	    hidden_size_1=5,
	    hidden_size_2=5,
	    output_size=1,
        activation='linear',
	    loss=BinaryCrossEntropy()
    )
    print(model)

    optimizer = SGD(model.parameters(), lr=.01)

    loss = []
    acc = []
    for i in range(5000):
        y_hat = model(x_train)
        loss.append(model.loss(y_hat, t=y_train))
        acc.append(accuracy(y_hat, y_train))

        model.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"epoch: {i}, loss: {loss[-1]}, accuracy: {acc[-1]}")

    y_pred = model(x_test)
    show_loss(loss, acc, 'loss.png')
    show_result(x_test, y_test, y_pred>0.5, 'result.png')
    

if __name__ == '__main__':
    main()