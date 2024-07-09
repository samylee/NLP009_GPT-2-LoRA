import argparse
import json

import encoder


parser = argparse.ArgumentParser()
parser.add_argument('--input', default='data/train.jsonl', type=str, help='ft input file')
parser.add_argument('--vocab', type=str, default='vocab', help='vocab path')
parser.add_argument('--output', default='data/train_encode.jsonl', type=str, help='ft output file')
parser.add_argument('--add_bos', default=True, help='')
parser.add_argument('--add_eos', default=True, help='')
args = parser.parse_args()


if __name__ == "__main__":
    enc = encoder.get_encoder(args.vocab)
    
    writer = open(args.output, 'w')

    with open(args.input, 'r') as reader:
        line_idx = 0
        for line in reader:
            items = json.loads(line.strip())
            context = items['context']
            completion = items['completion']

            bos = 50256
            eos = 50256
            context_bpes, _ = enc.encode(context) 
            context_bpes += [bos] if args.add_bos else []

            completion_bpes, _ = enc.encode(' ' + completion)
            completion_bpes += [eos] if args.add_eos else []

            ft_json = {}
            ft_json['context'] = context_bpes
            ft_json['completion'] = completion_bpes 
            writer.write(json.dumps(ft_json)+'\n')

            line_idx += 1

    writer.close()
