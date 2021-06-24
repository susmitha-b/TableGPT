import os
import json
import copy
import random
import codecs
import nltk
import transformers
from tqdm import tqdm

MAX_LEN = 512
BOS = 50257

def build_sentences(dataset_fold, data_type):
    with codecs.open('%s/%s.json'%(dataset_fold, data_type), 'r', 'utf-8') as fr:
        tables = []
        summs = []
        for line in tqdm(fr.readlines()):
            dic = json.loads(line)
            src_sents = []
            for key, content in dic.items():
                if key == 'Sentences':
                    content[0] = content[0].replace("-lrb-", "(")
                    content[0] = content[0].replace("-rrb-", ")")
                    tgt_sent = content[0].split()
                elif key != 'KB_id_tuples' and key != 'KB_str_tuples':
                    if content == '<none>' or content == '':
                        continue
                    attribute = [w for w in key.split('_') if w != '']
                    if len(attribute) == 0:
                        continue
                    content = content.replace("-lrb-", "(")
                    content = content.replace("-rrb-", ")")
                    entity = content.split()
                    src_sent = attribute + ['is'] + entity + ['.']
                    src_sents.extend(src_sent)
            tables.append(' '.join(src_sents))
            summs.append(' '.join(tgt_sent))
    return tables, summs
            

def build_src_tgt_corpus(dataset_fold, data_type):
    with codecs.open('%s/%s.json'%(dataset_fold, data_type), 'r', 'utf-8') as fr:
        test_src_corpus = []
        test_tgt_corpus = []
        overlength_cnt = 0
        for line in tqdm(fr.readlines()):
            dic = json.loads(line)
            src_sents = []
            for key, content in dic.items():
                if key == 'Sentences':
                    content[0] = content[0].replace("-lrb-", "(")
                    content[0] = content[0].replace("-rrb-", ")")
                    test_tgt = ' '.join(content[0].split())
                elif key != 'KB_id_tuples' and key != 'KB_str_tuples':
                    if content == '<none>' or content == '':
                        continue
                    attribute = key.split('_')
                    content = content.replace("-lrb-", "(")
                    content = content.replace("-rrb-", ")")
                    entity = content.split()
                    src_sent = attribute + ['is'] + entity + ['.']
                    src_sents.extend(src_sent)
            if len(src_sents) > MAX_LEN:
                overlength_cnt += 1
                continue
            src_sents.append('<table2text>')
            test_src_corpus.append(' '.join(src_sents))
            test_tgt_corpus.append(test_tgt)
    print('over length cnt: %d'%overlength_cnt)
    with codecs.open('%s/src_%s.txt'%(dataset_fold, data_type), 'w', 'utf-8') as fw:
        fw.write('\n'.join(test_src_corpus))
    with codecs.open('%s/tgt_%s.txt'%(dataset_fold, data_type), 'w', 'utf-8') as fw:
        fw.write('\n'.join(test_tgt_corpus))

def build_tokenized_files(dataset_fold, data_type, tables, summs):
    tokenize_ids_lst = []
    full_tokenizer = transformers.GPT2Tokenizer.from_pretrained('../models/gpt2/')
    full_tokenizer.add_tokens(['<table2text>'])
    overlength_cnt = 0
    for i in tqdm(range(len(tables))):
        tokenize_ids = full_tokenizer.encode(' '.join([tables[i], '<table2text>', summs[i], '<|endoftext|>']))
        if len(tokenize_ids) <= MAX_LEN:
            tokenize_ids_lst.append([' '.join([str(id) for id in tokenize_ids]), len(tokenize_ids)])
        else:    
            overlength_cnt += 1
    print('over length cnt: %d'%overlength_cnt)
    sorted_token_ids2lengths_lst = sorted(tokenize_ids_lst, key=lambda x: x[1])
    sorted_token_ids_lst = [tup[0] for tup in sorted_token_ids2lengths_lst]
    with open('%s/tokenized_%s.txt'%(dataset_fold, data_type), 'w') as f:
        f.write('\n'.join(sorted_token_ids_lst))
    print('finish')

def build_content_select_tokenized_files(dataset_fold, data_type, tables, summs):
    with codecs.open('%s/%s_content_select.txt'%(dataset_fold, data_type), 'r', 'utf-8') as fr:
        entity_lst = [line.strip() for line in fr.readlines()]
    tokenize_ids_lst = []
    full_tokenizer = transformers.GPT2Tokenizer.from_pretrained('../models/gpt2/')
    full_tokenizer.add_tokens(['<table2text>'])
    full_tokenizer.add_tokens(['<content_select>'])
    overlength_cnt = 0
    for i in tqdm(range(len(tables))):
        table2text_tokenize_ids = full_tokenizer.encode(' '.join([tables[i], '<table2text>', summs[i], '<|endoftext|>']))
        content_select_tokenize_ids = full_tokenizer.encode(' '.join([tables[i], '<content_select>', entity_lst[i], '<|endoftext|>']))
        if len(table2text_tokenize_ids) <= MAX_LEN and len(content_select_tokenize_ids) < MAX_LEN * 2:
            tokenize_ids_lst.append([' '.join([str(id) for id in table2text_tokenize_ids]), len(table2text_tokenize_ids), ' '.join([str(id) for id in content_select_tokenize_ids])])
        else:    
            overlength_cnt += 1
    print('over length cnt: %d'%overlength_cnt)
    sorted_token_ids2lengths_lst = sorted(tokenize_ids_lst, key=lambda x: x[1])
    sorted_table2text_token_ids_lst = [tup[0] for tup in sorted_token_ids2lengths_lst]
    sorted_content_select_token_ids_lst = [tup[2] for tup in sorted_token_ids2lengths_lst]
    with open('%s/table2text_tokenized_%s.txt'%(dataset_fold, data_type), 'w') as f:
        f.write('\n'.join(sorted_table2text_token_ids_lst))
    with open('%s/content_select_tokenized_%s.txt'%(dataset_fold, data_type), 'w') as f:
        f.write('\n'.join(sorted_content_select_token_ids_lst))
    print('finish')

def build_field2word(dataset_fold):
    reserve_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'CD', 'FW']
    with codecs.open('%s/train.json'%dataset_fold, 'r', 'utf-8') as fr:
        dict_lst = [json.loads(line) for line in fr.readlines()]
    field2word = {}
    for dic in tqdm(dict_lst):
        for key, value in dic.items():
            if key not in ['Sentences', 'KB_id_tuples', 'KB_str_tuples'] and (value != '<none>' and value != ''):
                attribute = '_'.join([w for w in key.split('_') if w != ''])
                if len(attribute) == 0:
                    continue
                for w, t in nltk.pos_tag(value.split()):
                    if t not in reserve_tags:
                        continue
                    if attribute not in field2word:
                        field2word[attribute] = set([w])
                    else:
                        field2word[attribute].add(w) 
    for k, v in field2word.items():
        field2word[k] = list(v)
    with codecs.open('%s/attribute2token.json'%dataset_fold, 'w') as fw:
        json.dump(field2word, fw)
    print('build field2word finish')

def build_replace_classify_tokenized_files(dataset_fold, tables, summs, pseudo_num, replace_entity_num):
    full_tokenizer = transformers.GPT2Tokenizer.from_pretrained('../models/gpt2/')
    full_tokenizer.add_tokens(['<table2text>'])
    with codecs.open('%s/train_summary_attribute.txt'%dataset_fold, 'r', 'utf-8') as fr:
        field_lst = [line.split() for line in fr.readlines()]
    with codecs.open('%s/attribute2token.json'%dataset_fold, 'r') as fr:
        field2token = json.load(fr)
    pseudo_token_ids_lst = [[] for _ in range(pseudo_num)]
    overlength_cnt = 0
    for t_i in tqdm(range(len(tables))):
        table_ws = tables[t_i].split()
        summ_ws = summs[t_i].split()
        field2idx = {}
        for f_i in range(len(field_lst[t_i])):
            field = field_lst[t_i][f_i]
            if field not in field2token.keys():
                continue
            if field not in field2idx:
                field2idx[field] = [f_i]
            else:
                field2idx[field].append(f_i)
        sent_ws = table_ws + ['<table2text>'] + summ_ws + ['<|endoftext|>']
        sent_ids = full_tokenizer.encode(' '.join(sent_ws))
        if len(sent_ids) > MAX_LEN:
            overlength_cnt += 1
            continue
        unchange_cnt = 0
        for p_i in range(pseudo_num):
            new_summ_ws = copy.deepcopy(summ_ws)
            # 0 represent token is original, 1 represent token is replaced
            replace_flags = [0 for _ in range(len(new_summ_ws))]
            if len(field2idx) != 0:
                if len(field2idx) == 1:
                    field_names = list(field2idx.keys())
                else:
                    field_names = random.sample(list(field2idx.keys()), replace_entity_num)
                for field in field_names:
                    sample_flag = True
                    sample_step = 0
                    while sample_flag:
                        sample_flag = False
                        if len(list(field2token[field])) == 1:
                            token_ids = list(field2token[field])
                            break
                        token_ids = random.sample(list(field2token[field]), 1)
                        for t_id in token_ids:
                            if t_id in summ_ws:
                                sample_flag = True
                        sample_step += 1
                        if sample_step >= 10:
                            break
                    if len(field2idx[field]) == 1:
                        replace_idxs = field2idx[field]
                    else:
                        replace_idxs = random.sample(field2idx[field], 1)
                    for idx, t_id in zip(replace_idxs, token_ids):
                        new_summ_ws[idx] = t_id
                        replace_flags[idx] = 1
            else:
                print(field_lst[t_i])
                unchange_cnt += 1
            replace_flags = [0 for _ in range(len(table_ws) + 1)] + replace_flags + [0]
            new_sent_ws = table_ws + ['<table2text>'] + new_summ_ws + ['<|endoftext|>']
            assert len(replace_flags) == len(new_sent_ws)
            new_sent_ids = []
            replace_ids_flags = []
            w_i = 0
            for w, r_f in zip(new_sent_ws, replace_flags):
                if w_i != 0:
                    w = ' ' + w
                w_i += 1
                w_ids = full_tokenizer.encode(w)
                new_sent_ids.extend(w_ids)
                replace_ids_flags.extend([r_f for _ in w_ids])
            assert len(replace_ids_flags) == len(new_sent_ids)
            pseudo_token_ids_lst[p_i].append([' '.join([str(id) for id in new_sent_ids]), ' '.join([str(id) for id in sent_ids]), ' '.join(str(flag) for flag in replace_ids_flags), len(sent_ids)])
    print('unchange cnt %d'%unchange_cnt)
    sorted_token_ids2lengths_lst = []
    for pseudo_token_ids in pseudo_token_ids_lst:
        sorted_token_ids2lengths_lst.extend(sorted(pseudo_token_ids, key=lambda x: x[3]))
    print(len(sorted_token_ids2lengths_lst))
    with open('%s/pseudo_tokenized_train.txt'%dataset_fold, 'w') as f:
        f.write('\n'.join([tup[0] for tup in sorted_token_ids2lengths_lst]))
    with open('%s/gold_tokenized_train.txt'%dataset_fold, 'w') as f:
        f.write('\n'.join([tup[1] for tup in sorted_token_ids2lengths_lst]))
    with open('%s/replace_flag_train.txt'%dataset_fold, 'w') as f:
        f.write('\n'.join([tup[2] for tup in sorted_token_ids2lengths_lst]))
    print('finish')

def build_entity_swap_mask(data_fold):
    with open('%s/pseudo_tokenized_train.txt'%data_fold, 'r') as fr:
        mask_lst = []
        for line in fr.readlines():
            ids = [int(token_id) for token_id in line.split()]
            bos_index = ids.index(BOS)
            masks = ['0'] * (bos_index + 1) + ['1'] * (len(ids) - bos_index - 1)
            mask_lst.append(' '.join(masks))
    with open('%s/replace_mask_train.txt'%data_fold, 'w') as f:
        f.write('\n'.join(mask_lst))


def merge_tokenized_files():
    file_names = ["Automobile", "Military_conflict", "Single", "Station", "UK_school", "Australian_place"]
    train_tokenize_ids_lst = []
    dev_tokenize_ids_lst = []
    for file_name in file_names:
        with codecs.open('%s/%s_tokenized_%s.txt'%(file_name, file_name.lower(), 'train'), 'r', 'utf-8') as fr:
            train_tokenize_ids_lst.extend([line.strip() for line in fr.readlines()])
        if os.path.exists('%s/%s_tokenized_%s.txt'%(file_name, file_name.lower(), 'dev')):
            with open('%s/%s_tokenized_%s.txt'%(file_name, file_name.lower(), 'dev')) as fr:
                dev_tokenize_ids_lst.extend([line.strip() for line in fr.readlines()])
        print('%s finish'%file_name)
    print('number of train %d'%len(train_tokenize_ids_lst))
    print('number of dev %d'%len(dev_tokenize_ids_lst))
    with codecs.open('small_domain_tokenized_train.txt', 'w', 'utf-8') as fw:
        fw.write('\n'.join(train_tokenize_ids_lst))
    with codecs.open('small_domain_tokenized_dev.txt', 'w', 'utf-8') as fw:
        fw.write('\n'.join(dev_tokenize_ids_lst))


def build_attribute_label(data_fold):
    full_tokenizer = transformers.GPT2Tokenizer.from_pretrained('../models/gpt2/')
    full_tokenizer.add_tokens(['<table2text>'])
    with codecs.open('%s/train.json'%(data_fold), 'r', 'utf-8') as fr:
        tables = []
        summarys = []
        fields = {}
        field_id = 1
        for line in tqdm(fr.readlines()):
            dic = json.loads(line)
            table = []
            for key, content in dic.items():
                if key == 'Sentences':
                    content[0] = content[0].replace("-lrb-", "(")
                    content[0] = content[0].replace("-rrb-", ")")
                    tgt_sent = content[0].split()
                elif key != 'KB_id_tuples' and key != 'KB_str_tuples':
                    if content == '<none>' or content == '':
                        continue
                    attribute = ' '.join([w for w in key.split('_') if w != ''])
                    if attribute == '':
                        continue
                    content = content.replace("-lrb-", "(")
                    content = content.replace("-rrb-", ")")
                    entity = content.split()
                    table.append([attribute, entity])
                    if attribute not in fields:
                        fields[attribute] = field_id
                        field_id += 1
            tables.append(table)
            summarys.append(' '.join(tgt_sent + ['<|endoftext|>']))
        print('field sequence label num: %d'%(field_id))

        train_tables, train_summs = build_sentences(data_fold, 'train')    
        
        field_label_lst = []
        label_mask_lst = []
        for i, table in tqdm(enumerate(tables)):
            summ = summarys[i]
            line = ' '.join([train_tables[i], '<table2text>', train_summs[i], '<|endoftext|>'])
            line_ids = full_tokenizer.encode(line)
            new_line_ids = []
            field_labels = []
            label_masks = []
            first_flag = True
            for field, entity in table:
                field_id = fields[field]
                if not first_flag:
                    field = ' ' + field
                else:
                    first_flag = False
                field_labels.extend(['0' for _ in full_tokenizer.encode(field)])
                label_masks.extend(['0' for _ in full_tokenizer.encode(field)])
                new_line_ids.extend(full_tokenizer.encode(field))
                field_labels.extend(['0' for _ in full_tokenizer.encode(' is')])
                label_masks.extend(['0' for _ in full_tokenizer.encode(' is')])
                new_line_ids.extend(full_tokenizer.encode(' is'))
                field_labels.extend([str(field_id) for _ in full_tokenizer.encode(' ' + ' '.join(entity))])
                label_masks.extend(['1' for _ in full_tokenizer.encode(' ' + ' '.join(entity))])
                new_line_ids.extend(full_tokenizer.encode(' ' + ' '.join(entity)))
                field_labels.extend(['0' for _ in full_tokenizer.encode(' .')])
                label_masks.extend(['0' for _ in full_tokenizer.encode(' .')])
                new_line_ids.extend(full_tokenizer.encode(' .'))
            # label_masks = ['1' for _ in field_labels]
            field_labels.extend(['0' for _ in full_tokenizer.encode(' '.join([' <table2text>', summ]))])
            new_line_ids.extend(full_tokenizer.encode(' '.join([' <table2text>', summ])))
            label_masks += ['0' for _ in full_tokenizer.encode(' '.join([' <table2text>', summ]))]
            try:
                assert new_line_ids == line_ids
                assert len(label_masks) == len(field_labels) and len(line_ids) == len(field_labels)
            except:
                print('train line %d'%i)
                print(len(line_ids), line_ids)
                print(len(new_line_ids), new_line_ids)
            field_label_lst.append(field_labels)
            label_mask_lst.append(label_masks)
    
    token_ids_lst = []
    overlength_cnt = 0
    for i in tqdm(range(len(train_tables))):
        field_labels = field_label_lst[i]
        label_masks = label_mask_lst[i]
        line = ' '.join([train_tables[i], '<table2text>', train_summs[i], '<|endoftext|>'])
        line_ids = full_tokenizer.encode(line)
        if len(line_ids) <= MAX_LEN:
            token_ids_lst.append([' '.join([str(id) for id in line_ids]), ' '.join(field_labels), ' '.join(label_masks), len(line_ids)])
        else:    
            overlength_cnt += 1
    print('%s dataset over length cnt: %d'%(data_fold, overlength_cnt))
    sorted_token_ids2lengths_lst = sorted(token_ids_lst, key=lambda x: x[3])
    sorted_field_label_lst = [tup[1] for tup in sorted_token_ids2lengths_lst]
    sorted_label_mask_lst = [tup[2] for tup in sorted_token_ids2lengths_lst]
    with open('%s/attribute_label_train.txt'%data_fold, 'w') as fw:
        fw.write('\n'.join(sorted_field_label_lst))
    with open('%s/label_mask_train.txt'%data_fold, 'w') as fw:
        fw.write('\n'.join(sorted_label_mask_lst))    
    print('finish')


if __name__ == '__main__':
    data_fold = 'WikiBio'
    # data_fold = 'Australian_place'
    # train_tables, train_summs = build_sentences(data_fold, 'train')
    # build_tokenized_files(data_fold, 'train', train_tables, train_summs)
    # build_src_tgt_corpus(data_fold, 'test')
    # build_src_tgt_corpus(data_fold, 'dev')
    build_attribute_label(data_fold)
    # build_src_tgt_corpus(data_fold, 'train')
    # build_content_select_tokenized_files(data_fold, 'train', train_tables, train_summs)
    # merge_tokenized_files()
    # build_field2word(data_fold)
    # build_replace_classify_tokenized_files(data_fold, train_tables, train_summs, 4, 2)
    # build_entity_swap_mask(data_fold)