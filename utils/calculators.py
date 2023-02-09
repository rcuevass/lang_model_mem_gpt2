import torch
import zlib
import numpy as np


def calculate_perplexity(sentence: str, model, tokenizer, device_tch_dev: torch.device) -> torch:
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device_tch_dev)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)


def compute_models_perplexity(lang_models_dict: dict, log_obj, texts, tokenizer, local_device: torch.device) -> tuple:
    scores_dict = {"XL": [], "S": [], "Lower": [], "zlib": []}
    samples = []
    model1 = lang_models_dict['model_1']
    model2 = lang_models_dict['model_2']
    log_obj.info('Looping over texts...')
    log_obj.info('Total number of samples in corpus = %s', len(texts))
    for text in texts:
        # perplexity of GPT2-XL and GPT2-S
        # log.info('Computing perplexities for text in turn... ')
        len_text_turn = len(text.split())
        log_obj.info('Computing perplexities for text in turn ...')
        log_obj.info('Length of  text in turn = %s ', str(len_text_turn))
        log_obj.info('First 50 characters of text = %s ', str(text)[:51])
        p1 = calculate_perplexity(text, model1, tokenizer, device_tch_dev=local_device)
        log_obj.info('Perplexity computed from model 1 = %s ', str(p1))
        p2 = calculate_perplexity(text, model2, tokenizer, device_tch_dev=local_device)
        log_obj.info('Perplexity computed from model 2 = %s ', str(p2))

        # perplexity on lower-case sample
        log_obj.info('Computing perplexity on lower case sample ...')
        p_lower = calculate_perplexity(text.lower(), model1, tokenizer, device_tch_dev=local_device)
        log_obj.info('Perplexity for lower case sample = %s', str(p_lower))

        log_obj.info('Computing Zlib entropy of text sample ...')
        zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
        log_obj.info('Zlib entropy of sample = %s', str(zlib_entropy))

        samples.append(text)
        scores_dict['XL'].append(p1)
        scores_dict['S'].append(p2)
        scores_dict['Lower'].append(p_lower)
        scores_dict['zlib'].append(zlib_entropy)

    scores_dict['XL'] = np.asarray(scores_dict['XL'])
    scores_dict['S'] = np.asarray(scores_dict['S'])
    scores_dict['Lower'] = np.asarray(scores_dict['Lower'])
    scores_dict['zlib'] = np.asarray(scores_dict['zlib'])

    return scores_dict, samples
