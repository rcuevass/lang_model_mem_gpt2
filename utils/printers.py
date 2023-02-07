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


def print_sorted_samples(log_object, scores_dict, samples):
    # Sort by perplexity
    log_object.info('Sorting by log perplexity...')
    metric = -np.log(scores_dict['XL'])
    log_object.info('======== top sample by XL perplexity ========')
    print_best(log_object, metric, samples, 'PPL', scores_dict['XL'])
    print()
    print()

    # Sort by ratio of log perplexities of S and XL models
    metric = np.log(scores_dict["S"]) / np.log(scores_dict["XL"])
    log_object.info('======== top sample by ratio of S and XL perplexities ========')
    print_best(log_object, metric, samples, "PPL-XL", scores_dict["XL"], "PPL-S", scores_dict["S"])
    print()
    print()

    # Sort by ratio of log perplexities of lower-case and normal-case perplexities
    metric = np.log(scores_dict["Lower"]) / np.log(scores_dict["XL"])
    log_object.info('======== top sample by ratio of lower-case and normal-case perplexities: ========')
    print_best(log_object, metric, samples, "PPL-XL", scores_dict["XL"], "PPL-XL-Lower", scores_dict["Lower"])
    print()
    print()

    # Sort by ratio of Zlib entropy and XL perplexity
    metric = scores_dict["zlib"] / np.log(scores_dict["XL"])
    log_object.info('======== top sample by ratio of Zlib entropy and XL perplexity: ========')
    print_best(log_object, metric, samples, 'PPL-XL', scores_dict['XL'], 'Zlib', scores_dict['zlib'])
