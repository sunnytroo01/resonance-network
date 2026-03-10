"""
Text generation for Resonance Network.

Supports:
- Greedy decoding
- Top-k sampling
- Top-p (nucleus) sampling
- Temperature scaling
- Repetition penalty
"""

import torch
import torch.nn.functional as F
import tiktoken


@torch.no_grad()
def generate(
    model,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    tokenizer_name: str = "gpt2",
    device: str = "cuda",
) -> str:
    """
    Generate text from a prompt using the Resonance Network.

    Args:
        model: trained ResonanceNetwork
        prompt: input text
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature (0 = greedy, higher = more random)
        top_k: top-k filtering (0 = disabled)
        top_p: nucleus sampling threshold
        repetition_penalty: penalty for repeating tokens
        tokenizer_name: tiktoken encoding name
        device: device to run on

    Returns:
        Generated text string
    """
    model.eval()
    enc = tiktoken.get_encoding(tokenizer_name)

    # Encode prompt
    input_ids = enc.encode(prompt, disallowed_special=())
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated = input_ids[0].tolist()

    for _ in range(max_new_tokens):
        # Truncate to max_seq_len
        context = input_ids[:, -model.max_seq_len:]

        # Forward pass
        logits, _, _ = model(context)
        next_logits = logits[0, -1, :]  # (vocab_size,)

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated):
                next_logits[token_id] /= repetition_penalty

        # Temperature
        if temperature > 0:
            next_logits = next_logits / temperature
        else:
            # Greedy
            next_token = next_logits.argmax().unsqueeze(0)
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            continue

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
            next_logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float('-inf')

        # Sample
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # Stop on EOS (tiktoken gpt2 doesn't have explicit EOS, but we can check)
        if next_token.item() == enc.eot_token:
            break

    return enc.decode(generated)
