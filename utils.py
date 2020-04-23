import torch
import torch.nn.functional as F


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, args, num_samples=1, temperature=1, stop_token=None, \
                    trigger=None, supress=None, repetition=None, top_k=0, top_p=0.0,  device='cuda'):
    if isinstance(context, list):
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0).repeat(num_samples, 1)

    generated = context
    batch_size = generated.shape[0]
    
    finished_template = [False for _ in range(batch_size)]
    finished_sentence = [False for _ in range(batch_size)]
    with torch.no_grad():
        for _ in range(length):
            outputs = model(generated, *args)
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            else:
                next_token_logits = outputs[:, -1, :] / (temperature if temperature > 0 else 1.)

            #next_token_logits[:, generated[-1].tolist()] /= repetition
            if repetition is not None:
                for b in range(batch_size):
                    if generated[:, -1][b].item() == repetition:
                        next_token_logits[b, repetition] = -float('Inf')

            if supress is not None:
                for b in range(batch_size):
                    if finished_template[b]:
                        next_token_logits[b, supress] = -float('Inf')

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

            if temperature == 0:  # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            
            if trigger:
                for b in range(batch_size):
                    if next_token[b].item() == trigger:
                        finished_template[b] = True
                    if next_token[b].item() == stop_token:
                        finished_sentence[b] = True

            generated = torch.cat((generated, next_token), dim=1)

            if all(finished_sentence):
                break

    return generated
