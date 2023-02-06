import torch


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
