import matplotlib.pyplot as plt


def plot_result(fname, x, y, pred_y):
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
    plt.close()


def plot_curve_by_exp_group(fname, title='', **kwargs):
    fig = plt.figure(figsize=(8, 4.5))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss & Accuracy')

    for label, data in kwargs.items():
        plt.plot(
            range(1, len(data)+1), data,
            '--' if 'accuracy' in label else '-',
            label=label
        )

    plt.legend(
        loc='best', bbox_to_anchor=(1.0, 1.0, 0.2, 0),
        fancybox=True, shadow=True
    )

    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
