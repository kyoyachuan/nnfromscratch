import matplotlib.pyplot as plt


def show_result(x, y, pred_y, fname):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.savefig(fname)


def show_loss(loss, acc, fname):
    plt.plot(loss, label='loss')
    plt.plot(acc, label='acc')
    plt.title('Loss-Epoch plot', fontsize=18)
    plt.xlabel('epoch')
    plt.ylabel('loss/acc')
    plt.ylim([-0.1, 1.1])
    plt.yticks([0.1 * i for i in range(11)])
    plt.legend()
    plt.grid()
    plt.savefig(fname)