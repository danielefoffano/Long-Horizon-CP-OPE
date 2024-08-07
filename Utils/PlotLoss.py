import matplotlib.pyplot as plt

def plot_loss(losses, training_steps):
    plt.plot(range(training_steps), losses)
    plt.show()