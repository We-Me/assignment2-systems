import argparse

import torch

from cs336_basics.bpe import BPETokenizer
from cs336_basics.functions import softmax_with_temperature
from cs336_basics.transformer_lm import TransformerLM


def top_p_sample(logits: torch.Tensor, top_p: float):
    q, idx = torch.sort(logits, descending=True)
    k = int((torch.cumsum(q, 0) < top_p).sum()) + 1   # 累计概率达到 p 的前 k 个
    q = q[:k]                                         # 截断
    u = torch.rand(1, device=q.device) * q.sum()      # 随机数换算到截断后的总概率
    pos = int((torch.cumsum(q, 0) < u).sum())         # u 落在哪个区间
    return int(idx[min(pos, k - 1)])


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    eos_token: str = '<|endoftext|>',
) -> str:
    eos_token_ids = tokenizer.encode(eos_token)
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) > context_length:
        seq = torch.tensor(prompt_ids[-context_length:], device=device, dtype=torch.long).unsqueeze(0)
    else:
        seq = torch.tensor(prompt_ids, device=device, dtype=torch.long).unsqueeze(0)

    model.eval()
    for _ in range(max_new_tokens):
        logits = model(seq)[0, -1]   # (vocab,) 最后一行的 logits
        logits = softmax_with_temperature(logits, dim=-1, t=temperature)
        prompt_ids.append(top_p_sample(logits, top_p))

        if prompt_ids[-(len(eos_token_ids)):] == eos_token_ids:
            break

        if len(prompt_ids) > context_length:
            seq = torch.tensor(prompt_ids[-context_length:], device=device, dtype=torch.long).unsqueeze(0)
        else:
            seq = torch.tensor(prompt_ids, device=device, dtype=torch.long).unsqueeze(0)

    complete = tokenizer.decode(prompt_ids)
    return prompt_ids, complete


def main(args: argparse.Namespace):
    seed=int(args.seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device)

    tokenizer = BPETokenizer.from_file(
        vocab_filepath=str(args.vocab_filepath),
        merges_filepath=str(args.merges_filepath),
        special_tokens=list(args.special_tokens)
    )

    model = TransformerLM(
        vocab_size=int(args.vocab_size),
        context_length=int(args.context_length),
        num_layers=int(args.num_layers),
        d_model=int(args.d_model),
        d_ff=int(args.d_ff),
        num_heads=int(args.num_heads),
        activation=str(args.activation),
        with_rope=(not bool(args.no_rope)),
        theta=float(args.theta),
        with_norm=(not bool(args.no_norm)),
        pre_norm=(not bool(args.post_norm)),
        eps=float(args.lm_eps),
    )
    state_dict = torch.load(str(args.checkpoint_file), map_location="cpu")
    model.load_state_dict(state_dict=state_dict["model"])
    model.to(device=device)

    prompt_ids, complete = generate(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=str(args.prompt),
        max_new_tokens=int(args.max_new_tokens),
        context_length=int(args.context_length),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        eos_token=str(args.eos_token),
    )
    print(f"total {len(prompt_ids)} tokens")
    print(complete)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('cs336 assignment1 basics generate')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--checkpoint_file', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, required=False, default=64)
    parser.add_argument('--temperature', type=float, required=False, default=1.0)
    parser.add_argument('--top_p', type=float, required=False, default=0.9)
    parser.add_argument('--eos_token', type=str, required=False, default='<|endoftext|>')
    # 分词器
    parser.add_argument('--vocab_filepath', type=str, required=True)
    parser.add_argument('--merges_filepath', type=str, required=True)
    parser.add_argument("--special_tokens", action="append", default=[])
    # 模型
    parser.add_argument('--vocab_size', type=int, required=True)
    parser.add_argument('--context_length', type=int, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--d_model', type=int, required=True)
    parser.add_argument('--d_ff', type=int, required=True)
    parser.add_argument('--num_heads', type=int, required=True)
    parser.add_argument('--activation', type=str, required=False, default="swiglu")
    parser.add_argument('--no_rope', action='store_true', default=False)
    parser.add_argument('--theta', type=float, required=False, default=10000.0)
    parser.add_argument('--no_norm', action='store_true', default=False)
    parser.add_argument('--post_norm', action='store_true', default=False)
    parser.add_argument('--lm_eps', type=float, required=False, default=1e-5)
    args = parser.parse_args()

    main(args)
