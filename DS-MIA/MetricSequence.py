import torch
import numpy as np

def createMetricSequences(targetX, targetY, num_metrics):
    privacyMetircs = np.split(targetX, num_metrics, axis=1)
    # for i in range(0, num_metrics):
    #     privacyMetircs[i] = np.roll(privacyMetircs[i], -1, axis=1)
    trajectory_data = []
    for i in range(0, len(targetX)):
        oneTr = np.vstack((privacyMetircs[0][i], privacyMetircs[1][i], privacyMetircs[2][i], privacyMetircs[3][i],
                           privacyMetircs[4][i]))  # (5*51)
        oneTr = np.swapaxes(oneTr, 0, 1)
        oneTr = np.insert(oneTr, 0, targetY[i], axis=1)
        trajectory_data.append(torch.Tensor(oneTr))

    return trajectory_data


def createLossTrajectories_Seq(targetX, targetY, num_metrics):
    privacyMetircs = targetX
    #privacyMetircs = np.roll(privacyMetircs, -1, axis=1)
    trajectory_data = []
    for i in range(0, len(targetX)):
        oneTr = privacyMetircs[i]
        oneTr = oneTr[:, np.newaxis]
        oneTr = np.insert(oneTr, 0, targetY[i], axis=1)
        trajectory_data.append(torch.Tensor(oneTr))

    return trajectory_data
