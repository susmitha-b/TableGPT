import os
import json
import codecs
import random
from tqdm import tqdm

def match_entity(entity_ws, word2index, min_match_len):
    cur_pos = None 
    match_len = 0
    for word in entity_ws:
        if word not in word2index:
            continue
        if cur_pos is None:
            cur_pos = word2index[word]
            match_len += 1
        else:
            new_cur_pos = []
            for idx in cur_pos:
                for pos in word2index[word]:
                    if pos == idx + 1:
                        new_cur_pos.append(pos)
            if len(new_cur_pos) == 0:
                break
            else:
                cur_pos = new_cur_pos
                match_len += 1
    if cur_pos is not None:
        if len(entity_ws) == 1 or match_len >= min_match_len:
            return [pos - match_len + 1 for pos in cur_pos], match_len
        else:
            return -1, 0
    else:
        return -1, 0

def extract_train_entity(dataset_fold):
    with codecs.open('%s/train.json'%dataset_fold, 'r', 'utf-8') as fr:
        train_entity_lst = []
        no_match_cnt = 0
        match_cnt = 0
        entity_cnt = 0
        for line in tqdm(fr.readlines()):
            dic = json.loads(line)
            tgt_sent = dic['Sentences'][0]
            tgt_sent = tgt_sent.replace("-lrb-", "(")
            tgt_sent = tgt_sent.replace("-rrb-", ")")
            tgt_sent = tgt_sent.split()
            word2index = {}
            for w_i, word in enumerate(tgt_sent):
                if word not in word2index:
                    word2index[word] = [w_i]
                else:
                    word2index[word].append(w_i)
            index2entity = {}
            for key, content in dic.items():
                if key not in ['KB_id_tuples', 'KB_str_tuples', 'Sentences']:
                    key = '_'.join([w for w in key.split('_') if w != ''])
                    if len(key) == 0:
                        continue
                    if content == '<none>' or content == '':
                        continue
                    content = content.replace("-lrb-", "(")
                    content = content.replace("-rrb-", ")")
                    entity = content.split()
                    entity_idxs, match_len = match_entity(entity, word2index, 2)
                    if entity_idxs == -1:
                        continue
                    for idx in entity_idxs:
                        if idx not in index2entity:
                            index2entity[idx] = [(match_len, key)]
                        else:
                            index2entity[idx].append((match_len, key))
            
            if len(index2entity) == 0:
                for key, content in dic.items():
                    if key not in ['KB_id_tuples', 'KB_str_tuples', 'Sentences']:
                        key = '_'.join([w for w in key.split('_') if w != ''])
                        if len(key) == 0:
                            continue
                        if content == '<none>' or content == '':
                            continue
                        content = content.replace("-lrb-", "(")
                        content = content.replace("-rrb-", ")")
                        entity = content.split()
                        entity_idxs, match_len = match_entity(entity, word2index, 1)
                        if entity_idxs == -1:
                            continue
                        for idx in entity_idxs:
                            if idx not in index2entity:
                                index2entity[idx] = [(match_len, key)]
                            else:
                                index2entity[idx].append((match_len, key))
            else:
                match_cnt += 1
                entity_cnt += len(index2entity)
            
            entity_sents = ['None' for _ in tgt_sent]
            if len(index2entity) != 0:
                sorted_index2entity = sorted(index2entity.items(), key=lambda x: x[0])
                for idx, tup_lst in sorted_index2entity:
                    max_len = tup_lst[0][0]
                    ent_sent = tup_lst[0][1]
                    for tup in tup_lst:
                        if tup[0] > max_len:
                            ent_sent = tup[1]
                            max_len = tup[0]
                    for e_len in range(max_len):
                        entity_sents[idx + e_len] = ent_sent
            else:
                no_match_cnt += 1
            train_entity_lst.append(' '.join(entity_sents))
    print('no match cnt is %d'%no_match_cnt)
    print('avg match entity is %f'%(entity_cnt/match_cnt))
    with codecs.open('%s/train_summary_attribute.txt'%dataset_fold, 'w', 'utf-8') as fw:
        fw.write('\n'.join(train_entity_lst))

def build_content_select(dataset_fold):
    with codecs.open('%s/train.json'%dataset_fold, 'r', 'utf-8') as fr:
        train_entity_lst = []
        no_match_cnt = 0
        match_cnt = 0
        entity_cnt = 0
        for line in tqdm(fr.readlines()):
            dic = json.loads(line)
            tgt_sent = dic['Sentences'][0].split()
            word2index = {}
            for w_i, word in enumerate(tgt_sent):
                if word not in word2index:
                    word2index[word] = [w_i]
                else:
                    word2index[word].append(w_i)
            index2entity = {}
            for key, content in dic.items():
                if key not in ['KB_id_tuples', 'KB_str_tuples', 'Sentences']:
                    if content == '<none>' or content == '':
                        continue
                    attribute = key.split('_')
                    entity = content.split()
                    src_sent = ' '.join(attribute + ['is'] + entity + ['.'])
                    entity_idxs, match_len = match_entity(entity, word2index, 2)
                    if entity_idxs == -1:
                        continue
                    for idx in entity_idxs:
                        if idx not in index2entity:
                            index2entity[idx] = [(match_len, src_sent)]
                        else:
                            index2entity[idx].append((match_len, src_sent))
            
            if len(index2entity) == 0:
                for key, content in dic.items():
                    if key not in ['KB_id_tuples', 'KB_str_tuples', 'Sentences']:
                        if content == '<none>' or content == '':
                            continue
                        attribute = key.split('_')
                        entity = content.split()
                        src_sent = ' '.join(attribute + ['is'] + entity + ['.'])
                        entity_idxs, match_len = match_entity(entity, word2index, 1)
                        if entity_idxs == -1:
                            continue
                        for idx in entity_idxs:
                            if idx not in index2entity:
                                index2entity[idx] = [(match_len, src_sent)]
                            else:
                                index2entity[idx].append((match_len, src_sent))
            else:
                match_cnt += 1
                entity_cnt += len(index2entity)
            
            entity_sents = []
            if len(index2entity) == 0:
                for key, content in dic.items():
                    if key not in ['KB_id_tuples', 'KB_str_tuples', 'Sentences']:
                        if content == '<none>' or content == '':
                            continue
                        attribute = key.split('_')
                        entity = content.split()
                        src_sent = ' '.join(attribute + ['is'] + entity + ['.'])
                        entity_sents.append(src_sent)
                entity_sents = random.sample(entity_sents, len(entity_sents) // 2)
                no_match_cnt += 1
            else:
                sorted_index2entity = sorted(index2entity.items(), key=lambda x: x[0])
                for idx, tup_lst in sorted_index2entity:
                    max_len = tup_lst[0][0]
                    ent_sent = tup_lst[0][1]
                    for tup in tup_lst:
                        if tup[0] > max_len:
                            ent_sent = tup[1]
                            max_len = tup[0]
                    entity_sents.append(ent_sent)
            train_entity_lst.append(' '.join(entity_sents))
    print('no match cnt is %d'%no_match_cnt)
    print('avg match entity is %f'%(entity_cnt/match_cnt))
    with codecs.open('%s/train_content_select.txt'%dataset_fold, 'w', 'utf-8') as fw:
        fw.write('\n'.join(train_entity_lst))
    
if __name__ == '__main__':
    data_fold = 'WikiBio'
    # build_content_select(data_fold)
    # data_fold = 'Songs'
    extract_train_entity(data_fold)