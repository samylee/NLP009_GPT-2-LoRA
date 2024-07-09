import argparse
import torch
import torch.nn.functional as F

import encoder
from model import GPT2Config, GPT2LMModel


parser = argparse.ArgumentParser(description='PyTorch GPT2 ft script')
parser.add_argument('--device', type=int, default=0, help='gradient accumulation steps')
parser.add_argument('--model_path', type=str, default='trained_models/GPT2_S/e2e/model.26289.pt', help='gradient accumulation steps')
parser.add_argument('--vocab', type=str, default='vocab', help='vocab path')
parser.add_argument('--max_seq_len', type=int, default=512, help='gradient accumulation steps')
parser.add_argument('--topk', type=int, default=10, help='gradient accumulation steps')
parser.add_argument('--lora_dim', type=int, default=4, help='lora attn dimension')
parser.add_argument('--lora_alpha', type=int, default=32, help='lora attn alpha')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='dropout probability for lora layers')
args = parser.parse_args()


def tok_k_logits(logits, k):
    v, ix = torch.topk(logits,k)
    out = logits.clone()
    out[out < v[:,[-1]]] = -float('inf')
    return out


def predict():
    # load vocab
    enc = encoder.get_encoder(args.vocab)

    # load model
    config = GPT2Config(
                n_embd=768, n_layer=12, n_head=12,
                lora_attn_dim=args.lora_dim,
                lora_attn_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )

    lm_net = GPT2LMModel(config)
    if args.model_path is not None:
        print('loading model.')
        state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        lm_net.load_state_dict(state_dict)
    lm_net.eval()
    lm_net = lm_net.to(args.device)

    # input context
    context = 'name : Alimentum | area : city centre | family friendly : no | near : Burger King'
    print('Input:' + '\n' + context + '\n' + 'Output:')

    bos = 50256
    eos = 50256
    context_bpes, _ = enc.encode(context)
    context_bpes += [bos]
    for _ in range(args.max_seq_len - len(context_bpes)):
        input_token = torch.tensor([context_bpes]).to(args.device)
        # forward
        logits, _ = lm_net(input_token)
        logits = logits[:, -1, :]
        logits = tok_k_logits(logits, args.topk)
        # Forward to linear classify token in vocab and Softmax
        probs = F.softmax(logits, dim=-1)

        index = torch.multinomial(probs, num_samples=1)
        if index.item() == eos:
            break

        context_bpes.append(index.item())
        print(enc.decode([index.item()]), end='')

if __name__ == '__main__':
    predict()