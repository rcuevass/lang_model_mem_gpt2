"""
Generate samples with GPT-2 and filter out those that are likely to be
memorized samples from the training set.
"""

import logging
import time
from utils.logging import get_log_object
from utils.calculators import compute_models_perplexity
from utils.parsers import parse_arguments, parse_commoncrawl
from utils.printers import print_sorted_samples
import numpy as np
import sys
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

logging.basicConfig(level='ERROR')
log = get_log_object()

log.info('Detecting local device...')
local_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(top_k_int: int = 40, seq_len_int: int = 256, gpt2_size_str: str = 'gpt2-medium'):
    """
    seq_len_int - number of tokens to generate
    top_k_int - sample from the top_k tokens output by the model
    gpt2_size_str - size of GPT2 to load. Options: gpt2-medium, gpt2-xl

    """
    log.info('Using device = %s', local_device)
    print(f"using device: {local_device}")

    if args.internet_sampling:
        print("Loading common crawl...")
        cc = parse_commoncrawl(args.wet_file)

    log.info('Getting GPT2 tokenizer...')
    t1 = time.time()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    t2 = time.time()
    rounded_time = round(t2 - t1, 2)
    log.info('GPT2 loaded in = %s seconds', str(rounded_time))

    log.info('Selecting padding side...')
    tokenizer.padding_side = "left"
    log.info('Padding side set to = %s', tokenizer.padding_side)
    tokenizer.pad_token = tokenizer.eos_token

    log.info('Getting pre-trained GPT2 model. Size = %s', gpt2_size_str)
    t1 = time.time()
    model1 = GPT2LMHeadModel.from_pretrained(gpt2_size_str, return_dict=True).to(local_device)

    t2 = time.time()
    rounded_time = round(t2 - t1, 2)
    log.info('GPT2 of size =%s loaded in = %s seconds', gpt2_size_str, str(rounded_time))

    model1.config.pad_token_id = model1.config.eos_token_id

    log.info('Getting pre-trained GPT2 model...')
    t1 = time.time()
    model2 = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True).to(local_device)
    t2 = time.time()
    rounded_time = round(t2 - t1, 2)
    log.info('GPT2 loaded in = %s seconds', str(rounded_time))

    model1.eval()
    model2.eval()

    num_batches = int(np.ceil(args.N / args.batch_size))
    log.info('Number of batches to be used = %s', str(num_batches))

    # Set up progress bar based on number of samples to generate...
    with tqdm(total=args.N) as pbar:

        # loop over number of batches to...
        log.info(' Looping over %s batches...', num_batches)
        log.info('Internet sampling set to = %s', str(args.internet_sampling))
        for i in range(num_batches):
            # ... encode the prompts ...
            if args.internet_sampling:
                # pick a random 10-token prompt in common crawl
                log.info('Internet sampling detected...')
                input_len = 10
                input_ids = []
                attention_mask = []

                while len(input_ids) < args.batch_size:
                    # take some random words in common crawl
                    r = np.random.randint(0, len(cc))
                    log.info('Using the following prompt ... ')
                    prompt = " ".join(cc[r:r + 100].split(" ")[1:-1])
                    log.info('prompt = %s', prompt)

                    # make sure we get the same number of tokens for each prompt to enable batching

                    inputs = tokenizer(prompt, return_tensors="pt", max_length=input_len, truncation=True)
                    num_inp_tok = len(inputs['input_ids'][0])
                    log.info('Number of inputs from tokenizer = %s', str(num_inp_tok))
                    if num_inp_tok == input_len:
                        input_ids.append(inputs['input_ids'][0])
                        attention_mask.append(inputs['attention_mask'][0])

                inputs = {'input_ids': torch.stack(input_ids),
                          'attention_mask': torch.stack(attention_mask)}

                # the actual truncated prompts
                prompts = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            else:
                log.info('No internet sampling detected...')
                log.info('Batch size from argument = %s', str(args.batch_size))
                prompts = ["<|endoftext|>"] * args.batch_size
                log.info('Prompts used = %s', str(prompts))

                input_len = 1
                log.info('Getting inputs from prompts tokenization ... ')
                inputs = tokenizer(prompts, return_tensors="pt", padding=True)
                log.info('Inputs obtained from tokenizer ...')

            # batch generation
            log.info('Performing batch generation ...')

            output_sequences = model1.generate(
                input_ids=inputs['input_ids'].to(local_device),
                attention_mask=inputs['attention_mask'].to(local_device),
                max_length=input_len + seq_len_int,
                do_sample=True,
                top_k=top_k_int,
                top_p=1.0
            )
            log.info('Batch generation completed ...')

            texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

            log.info('Looping over texts...')
            log.info('Total number of samples in corpus = %s', len(texts))

            lang_models_dict = {'model_1': model1, 'model_2': model2}
            scores, samples = compute_models_perplexity(lang_models_dict, log, texts, tokenizer, local_device)

            pbar.update(args.batch_size)

    print_sorted_samples(log, scores, samples)


if __name__ == '__main__':
    log.info('Parsing arguments from prompt ...')
    args = parse_arguments(sys.argv[1:])
    log.info('Arguments parsed = %s', str(args))

    log.info('Starting extraction...')
    time_1 = time.time()
    main()
    time_2 = time.time()
    rounded_total_time = round(time_2 - time_1, 2)
    rounded_total_time_str = str(round(rounded_total_time/60, 1))
    log.info('Total time of execution in = %s minutes', rounded_total_time_str)
