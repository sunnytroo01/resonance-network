"""
Interactive chat with a trained Resonance Network.

Usage:
    python chat.py --checkpoint checkpoints/checkpoint_100000.pt
"""

import argparse
import torch
from resonance.model.resonance_network import ResonanceNetwork
from resonance.generate import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    # Load checkpoint
    print("Loading model...")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt["config"]["model"]

    model = ResonanceNetwork(
        vocab_size=50257,  # tiktoken gpt2
        dim=config["dim"],
        n_layers=config["n_layers"],
        max_seq_len=config.get("max_seq_len", 2048),
        num_heads=config["num_heads"],
        coupling_rank=config.get("coupling_rank", config["dim"] // 4),
        num_stored_patterns=config.get("num_stored_patterns", 256),
        hopfield_steps=config.get("hopfield_steps", 1),
        mag_expansion=config.get("mag_expansion", 4),
        dropout=0.0,  # no dropout at inference
        dt=config.get("dt", 0.1),
        use_sparsemax=config.get("use_sparsemax", False),
        stability_weight=0.0,
    ).to(args.device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = model.get_num_params(False)
    print(f"Loaded {n_params:,} parameter Resonance Network")
    print(f"Type 'quit' to exit\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if prompt.lower() in ("quit", "exit", "q"):
            break

        if not prompt:
            continue

        output = generate(
            model,
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
        )

        # Print only the generated part (after prompt)
        response = output[len(prompt):]
        print(f"Model: {response}\n")


if __name__ == "__main__":
    main()
