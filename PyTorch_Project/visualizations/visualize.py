import matplotlib.pyplot as plt

def plot_loss(train_losses, validation_losses, path_output: str = './losses.png', xlab: str = 'x', ylab: str = 'y'):
    """Plot train and validation loss over epochs.
    """
    plt.style.use('seaborn')
    plt.plot(train_losses, label='Train loss (min: ' + str(round(min(train_losses), 3)) + ' at epoch ' + str(train_losses.index(min(train_losses))) + ')')
    plt.plot(validation_losses, label='Validation loss (min: ' + str(round(min(validation_losses), 3)) + ' at epoch ' + str(validation_losses.index(min(validation_losses))) + ')')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.text(0.1,0.9,'Hello World !')
    #plt.xlim([5, 9])
    save_plot(plt, path_output)

def save_plot(plt, path_output: str = './losses.png'):
    if(plt is not None):
        print('Saving plot to {0}'.format(path_output))
        plt.savefig(path_output)
    else:
        print('Can\'t save plot.')