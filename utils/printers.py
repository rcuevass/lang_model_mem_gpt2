import numpy as np
from pprint import pprint


def print_best(log_object, metric_lst: list, samples_lst: list, name1_str: str, scores1_np: np.array,
               name2_str: str = None, scores2_np: np.array = None, num_best_smpl: int = 10):
    """
    print the `n` best samples according to the given `metric`
    """
    idxs = np.argsort(metric_lst)[::-1][:num_best_smpl]

    for i, idx in enumerate(idxs):

        if scores2_np is not None:
            print(f"{i + 1}: {name1_str}={scores1_np[idx]:.3f}, {name2_str}={scores2_np[idx]:.3f},"
                  f"score={metric_lst[idx]:.3f}")
            log_object.info('%i : %s = %.3f, %s = %.3f, score = %.3f', i + 1, name1_str, scores1_np[idx],
                     name2_str, scores2_np[idx], metric_lst[idx])
        else:
            print(f"{i + 1}: {name1_str}={scores1_np[idx]:.3f}, , score={metric_lst[idx]:.3f}")

        print()
        pprint(samples_lst[idx])
        log_object.info('Sample %i in turn =  %s ', i + 1, str(samples_lst[idx]))
        print()
        print()
