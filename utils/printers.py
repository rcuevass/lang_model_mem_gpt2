import numpy as np
from pprint import pprint


def print_best(file_obj, log_object, metric_lst: list, samples_lst: list, name1_str: str, scores1_np: np.array,
               name2_str: str = None, scores2_np: np.array = None, num_best_smpl: int = 10):
    """
    print the `n` best samples according to the given `metric`
    """
    idxs = np.argsort(metric_lst)[::-1][:num_best_smpl]

    for i, idx in enumerate(idxs):

        if scores2_np is not None:
            print(f"{i + 1}: {name1_str}={scores1_np[idx]:.3f}, {name2_str}={scores2_np[idx]:.3f},"
                  f"score={metric_lst[idx]:.3f}")
            file_obj.write(f"{i + 1}: {name1_str}={scores1_np[idx]:.3f}, {name2_str}={scores2_np[idx]:.3f},"
                           f"score={metric_lst[idx]:.3f}" + "\n")
            log_object.info('%i : %s = %.3f, %s = %.3f, score = %.3f', i + 1, name1_str, scores1_np[idx],
                            name2_str, scores2_np[idx], metric_lst[idx])
        else:
            print(f"{i + 1}: {name1_str}={scores1_np[idx]:.3f}, , score={metric_lst[idx]:.3f}" + "\n")
            file_obj.write(f"{i + 1}: {name1_str}={scores1_np[idx]:.3f}, , score={metric_lst[idx]:.3f}")

        print()
        pprint(samples_lst[idx])
        file_obj.write(str(i + 1) + " " + str(samples_lst[idx]) + "\n")
        file_obj.write(" " + "\n")
        file_obj.write("===================================================================================" + "\n")
        file_obj.write(" " + "\n")
        log_object.info('Sample %i in turn =  %s ', i + 1, str(samples_lst[idx]))
        print()
        print()


def print_sorted_samples(log_object, scores_dict, samples,  file_name='output/extracted_samples.txt'):
    file_obj = open(file_name, 'w')
    # Sort by perplexity
    # See section 6.1 - Perplexity: the perplexity of the largest GPT-2 model
    log_object.info('Sorting by log perplexity...')
    metric = -np.log(scores_dict['XL'])
    log_object.info('======== top sample by XL perplexity ========')
    print_best(file_obj, log_object, metric, samples, 'PPL', scores_dict['XL'])
    print()
    print()

    # Sort by ratio of log perplexities of S and XL models
    # See section 6.1 - Small: the ratio of log-perplexities of the largest GPT-2
    # model and the Small GPT-2 model
    metric = np.log(scores_dict["S"]) / np.log(scores_dict["XL"])
    log_object.info('======== top sample by ratio of S and XL perplexities ========')
    print_best(file_obj, log_object, metric, samples, "PPL-XL", scores_dict["XL"], "PPL-S", scores_dict["S"])
    print()
    print()

    # Sort by ratio of log perplexities of lower-case and normal-case perplexities
    # See section 6.1 - Lowercase: the ratio of perplexities of the GPT-2 model
    # on the original sample and on the lower cased sample
    metric = np.log(scores_dict["Lower"]) / np.log(scores_dict["XL"])
    log_object.info('======== top sample by ratio of lower-case and normal-case perplexities: ========')
    print_best(file_obj, log_object, metric, samples, "PPL-XL", scores_dict["XL"], "PPL-XL-Lower", scores_dict["Lower"])
    print()
    print()

    # Sort by ratio of Zlib entropy and XL perplexity
    # See section 6.1 - zlib: the ratio of the (log) of the GPT-2 perplexity and the
    # zlib entropy (as computed by compressing the text).
    metric = scores_dict["zlib"] / np.log(scores_dict["XL"])
    log_object.info('======== top sample by ratio of Zlib entropy and XL perplexity: ========')
    print_best(file_obj, log_object, metric, samples, 'PPL-XL', scores_dict['XL'], 'Zlib', scores_dict['zlib'])

    file_obj.close()
