import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(num_games, 
                        scheduler_scores, 
                        worker_scores, 
                        plot_dir, 
                        env_name, 
                        file_type):
    """

    """

    scheduler_scores_running_avg = np.zeros(len(scheduler_scores))
    worker_scores_running_avg = np.zeros(len(worker_scores))

    for i in range(num_games):
        scheduler_scores_running_avg[i] = np.mean(scheduler_scores_running_avg[max(0, i-100): i+1])
        worker_scores_running_avg[i] = np.mean(worker_scores_running_avg[max(0, i-100): i+1])

    plt.plot(range(1,num_games+1), scheduler_scores_running_avg, label="Scheduler scores")
    plt.plot(range(1,num_games+1), worker_scores_running_avg, label="Worker scores")

    plt.title("{} running average of 100 scores".format(env_name))
    plt.savefig(plot_dir + env_name + file_type)


