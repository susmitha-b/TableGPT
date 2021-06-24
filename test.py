import torch
import torch.nn.functional as F
import os
import codecs
import argparse
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# from gpt2_model import GPT2LMHeadModel

BOS = 50257
# BOS = 50258
EOS = 50256
PAD_ID = 15636
MAX_LEN = 900

def rebuild_sent(line):
    ws = []
    for i, w in enumerate(line.split()):
        if w[-1] == ',':
            ws.append(w[:-1])
            ws.append(',')
        elif i == len(line.split()) - 1:
            if w[-1] == '.':
                ws.append(w[:-1])
                ws.append('.')
            else:
                ws.append(w)
        else:
            ws.append(w)
    return ' '.join(ws)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--max_length', default=50, type=int, required=False, help='生成文本最长长度')
    parser.add_argument('--num_beams', default=5, type=int, required=False, help='束搜索宽度')
    parser.add_argument('--tokenizer_path', default='models/gpt2/', type=str, required=False, help='词表路径')
    parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--src_test', default='data/WikiBio/wikibio_src_test.txt', type=str, required=False, help='测试输入文件路径')
    parser.add_argument('--save_output_path', default='.', type=str, required=False, help="保存样本的路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False)
    parser.add_argument('--length_penalty', default=1.0, type=float, required=False)
    parser.add_argument('--do_sample', action='store_true', help='生成时候是否对词表概率做采样')
    parser.add_argument('--eos', action='store_true', help='beam search生成时候是否遇到eos符号停止')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    max_length = args.max_length
    batch_size = args.batch_size
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    num_beams = args.num_beams
    repetition_penalty = args.repetition_penalty
    length_penalty = args.length_penalty
    do_sample = args.do_sample

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.add_tokens(['<table2text>'])
    # tokenizer.add_tokens(['<content_select>'])
    # tokenizer.add_tokens(['<rewrite>'])
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    with codecs.open(args.src_test, 'r', 'utf-8') as fr:
        # test_srcs = [' '.join(line.split() + ['<table2text>']) for line in fr.readlines()]
        test_srcs = [line.strip() for line in fr.readlines()]
    start_idx = 0
    if os.path.exists(args.save_output_path):
        with codecs.open(args.save_output_path, 'r', 'utf-8') as fr:
            start_idx = len(fr.readlines())

    # tgt_lst = []
    out_file = codecs.open(args.save_output_path, 'a', 'utf-8')
    with torch.no_grad():
        total_steps = len(test_srcs) // batch_size if len(test_srcs) % batch_size == 0 else len(test_srcs) // batch_size + 1
        for step in tqdm(range(total_steps)):
            if step * batch_size < start_idx:
                continue
            test_inputs = test_srcs[step * batch_size: (step + 1) * batch_size]
            input_ids = []
            for test_input in test_inputs:
                input_ids.append(tokenizer.encode(test_input))
            if len(input_ids[0]) > MAX_LEN:
                input_ids[0] = input_ids[0][:MAX_LEN] + [BOS]
                print('source input over max length')
            src_lengths = len(input_ids[0])
            batch_input = torch.tensor(input_ids).long().to(device)
            if args.eos:
                output = model.generate(input_ids=batch_input, do_sample=do_sample, max_length=src_lengths + max_length, num_beams=num_beams, bos_token_id=BOS, pad_token_id=PAD_ID, eos_token_ids=EOS, length_penalty=length_penalty)
            else:
                output = model.generate(input_ids=batch_input, do_sample=do_sample, max_length=src_lengths + max_length, num_beams=num_beams, bos_token_id=BOS, pad_token_id=PAD_ID, eos_token_ids=0, length_penalty=length_penalty)
            output_ids = output.tolist()[0]
            try:
                tgt_ids = output_ids[(output_ids.index(BOS) + 1): output_ids.index(EOS)]
            except:
                print("%d test generation over %d"%(step, max_length))
                if args.eos:
                    output = model.generate(input_ids=batch_input, do_sample=do_sample, max_length=src_lengths + max_length * 2, num_beams=num_beams, bos_token_id=BOS, pad_token_id=PAD_ID, eos_token_ids=EOS, length_penalty=length_penalty)
                else:
                    output = model.generate(input_ids=batch_input, do_sample=do_sample, max_length=src_lengths + max_length * 2, num_beams=num_beams, bos_token_id=BOS, pad_token_id=PAD_ID, eos_token_ids=0, length_penalty=length_penalty)
                output_ids = output.tolist()[0]
                try:
                    tgt_ids = output_ids[(output_ids.index(BOS) + 1): output_ids.index(EOS)]
                except:
                    tgt_ids = output_ids[(output_ids.index(BOS) + 1):]
                    print("%d test generation over %d"%(step, max_length * 2))
            tgt_ids = [i for i in tgt_ids if i != 50258]
            output_sent = rebuild_sent(tokenizer.decode(tgt_ids))
            out_file.write(output_sent + '\n')
            out_file.flush()
    out_file.close()

if __name__ == '__main__':
    main()