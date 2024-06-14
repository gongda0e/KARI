import os
import nltk
import os
import argparse
import pdb
import torch
import numpy as np
from nltk.parse.generate import generate
from nltk.grammar import CFG
import itertools
from itertools import permutations
from collections import Counter, defaultdict
import math
import copy

class SingleRule:
    def __init__(self, head, body_list, prob_list=None):
        self.head = head
        self.body_list = body_list
        self.prob_list = prob_list

    def __repr__(self):
        rep = ''
        rep += self.head
        rep += ' -> '
        for i, (body, prob) in enumerate(zip(self.body_list, self.prob_list)):
            rep += body + ' [' + str(prob) + ']'
            if i != len(self.body_list) - 1:
                rep += ' | '
        return rep

def remove_consecutive(l):
    l_ = []
    flag = False
    for a in l:
        for i in range(len(a) - 1):
            if a[i] == a[i+1]:
                flag = True
                break
        if not flag:
            l_.append(a)
        flag = False
    return l_

def concat_seqs_to_seqs(seqs1, seqs2, max_len=10):
    seqs = []
    for seq1 in seqs1:
        seqs += [' '.join([seq1, seq2]) for seq2 in seqs2 if len(seq1.split()) + len(seq2.split()) < max_len]
    seqs = list(set([' '.join(seq.rstrip().lstrip().split()) for seq in seqs]))
    return seqs

def state(s: str, n: int):
    return s + str(n)

def lindex(mylist, myvalue):
    return mylist.index(myvalue)

def rindex(mylist, myvalue):
    return len(mylist) - mylist[::-1].index(myvalue) - 1

def to_key_index(seqs, actions):
    ind_seqs = []
    actions = sum(actions, [])
    for seq in seqs:
        ind_seqs.append(tuple([actions.index(a) for a in seq]))
    return ind_seqs

def remain_key_actions(seqs, actions):
    key_list = []
    for seq in seqs:
        key = []
        for a in seq:
            if a in actions and a not in key:
                key.append(a)
        key_list.append(key)
    return key_list

def get_key_actions(seqs):
    num_seq = len(seqs)
    seqs = np.concatenate([np.unique(np.array(seq)) for seq in seqs], axis=0).tolist()
    count_dict = dict(Counter(seqs))
    key_actions = [k for k, v in count_dict.items() if v == num_seq]

    return key_actions

def split_seq_by_action(seq, actions):
    seqs = []
    prev_idx = 0
    flat_actions = sum(actions, [])
    flag_actions_1 = [False for _ in flat_actions]
    flag_actions_2 = [False for _ in flat_actions]

    for i in range(0, len(seq)):
        if seq[i] in flat_actions:
            idx = flat_actions.index(seq[i])
            if all(flag_actions_1):
                seqs.append(seq[prev_idx:i])
                prev_idx = i
                flag_actions_1 = [False for _ in flat_actions]
            flag_actions_1[idx] = True

    if len(seqs) == 0 or all(flag_actions_1):
        seqs.append(seq[prev_idx:])
    else:
        seqs[-1] += seq[prev_idx:]
    return seqs

def find_equivalent(action, equi_list):
    n_list = len(equi_list)
    flag = 0
    for equi_set in equi_list:
        if action in equi_set:
            return equi_set
    return None

def get_first_last_actions(seqs, key_actions, actions, actions2idx):
    first_map = torch.zeros(len(actions2idx)).bool()
    not_first_map = torch.zeros(len(actions2idx)).bool()
    last_map = torch.zeros(len(actions2idx)).bool()
    not_last_map = torch.zeros(len(actions2idx)).bool()
    fseqs = []
    lseqs = []
    mseqs = []

    for seq in seqs:
        most_left = len(seq)
        most_right = -1
        for k in key_actions:
            left, right = lindex(seq, k), rindex(seq, k)
            most_left = min(most_left, left)
            most_right = max(most_right, right)
        for i, a in enumerate(seq):
            if i < most_left:
                first_map[actions2idx[a]] = True
            else:
                not_first_map[actions2idx[a]] = True
            if i > most_right:
                last_map[actions2idx[a]] = True
            else:
                not_last_map[actions2idx[a]] = True
        fseqs.append(seq[:most_left])
        lseqs.append(seq[most_right+1:])
        mseqs.append(seq[most_left:most_right+1])

    first_map = first_map & ~not_first_map
    last_map = last_map & ~not_last_map
    firsts = actions[first_map]
    lasts = actions[last_map]

    not_lasts = actions[~last_map]
    lseqs_, mseqs_ = [], []
    for lseq, mseq in zip(lseqs, mseqs):
        if len(lasts) == 0:
            lseqs_.append([])
            mseqs_.append(mseq+lseq)
        else:
            idx = min([lseq.index(last) if last in lseq else 10000 for last in lasts])
            lseqs_.append(lseq[idx:])
            mseqs_.append(mseq+lseq[:idx])

    return fseqs, lseqs_, mseqs_

def get_tdmap(seqs, actions2idx):
    n_actions = len(actions2idx)
    tdmap = torch.zeros(n_actions, n_actions)
    for seq in seqs:
        for i in range(len(seq)):
            for j in range(i+1, len(seq)):
                tdmap[actions2idx[seq[i]], actions2idx[seq[j]]] = 1
    return tdmap

def select_tdmap(tdmap, actions2idx, actions):
    new_tdmap = []
    included = []
    # get row
    for action in actions:
        new_tdmap.append(tdmap[actions2idx[action]])
        included.append(actions2idx[action])
    new_tdmap = torch.stack(new_tdmap, 0)
    new_tdmap_t = torch.transpose(new_tdmap, 0, 1)
    new_tdmap2 = []
    for idx in range(len(actions2idx)):
        if idx in included:
            new_tdmap2.append(new_tdmap_t[idx])
    new_tdmap2 = torch.stack(new_tdmap2, 0)
    new_tdmap2 = torch.transpose(new_tdmap2, 0, 1)

    return new_tdmap2

def get_temporal_dependency(seqs, tdmap_ori, actions2idx_ori, actions, args):
    if len(actions) == 1:
        return [actions.tolist()]
    actions2idx = {a: i for i, a in enumerate(actions)}
    idx2actions = {i: a for i, a in enumerate(actions)}
    tdmap = torch.zeros(len(actions), len(actions))
    for seq in seqs:
        for i in range(len(seq)):
            for j in range(i+1, len(seq)):
                tdmap[actions2idx[seq[i]], actions2idx[seq[j]]] = 1

    tdmap_t = torch.transpose(tdmap, 0, 1)
    A = tdmap-tdmap_t
    A_zero = (A==0)
    A_zero_row = A_zero.sum(0)
    if len(tdmap) in A_zero_row:
        # if at least one row has all zeros, then every actions are temporally equivalent.
        return [actions.tolist()]

    # if every colum has zeros in the upper triangular zone, every action is equivalent
    lower_tri = np.tril(np.ones((len(tdmap), len(tdmap))))
    lower_tri_mask = torch.Tensor.bool(torch.Tensor((lower_tri == 1)))
    upper_A = copy.deepcopy(A).masked_fill_(lower_tri_mask, 1)
    upper_A_zero = (upper_A == 0)
    upper_A_col = upper_A_zero.sum(0)
    if 0 not in upper_A_col[1:]:
        # if every column has zeros, then every action is equivalent
        return [actions.tolist()]

    equi_list = []
    for idx in range(len(A)):
        equi_set = set((A[idx] == 0).nonzero(as_tuple=True)[0].numpy())
        if equi_set not in equi_list:
            equi_list.append(equi_set)

    merged_list = merge_sets(equi_list)

    #set temporal order between sets
    ordered_idx = [list(merged_list[0])]
    merged_list = merged_list[1:]
    for mdx in range(len(merged_list)):
        m_action = list(merged_list[mdx])[0]
        check_orders = []
        for odx in range(len(ordered_idx)):
            o_action = ordered_idx[odx][0]
            order = A[m_action][o_action]
            if order == -1:
                # action -> curr_action
                input_pos = odx+1
            elif order == 1:
                input_pos = odx
                break
            else:
                pdb.set_trace()
                print("order is zero. sth is wrong!")
        ordered_idx.insert(input_pos, list(merged_list[mdx]))

    #error occurs
    merged=[]
    for o_idx in range(len(ordered_idx)-1, 0, -1):
        c_actions = ordered_idx[o_idx]
        inserted = False
        for b_idx in range(o_idx-1, -1, -1):
            b_actions = ordered_idx[b_idx]
            inserted = False
            for c_action in c_actions:
                for b_action in b_actions:
                    order = A[c_action][b_action]
                    if order == 1:
                        #merge c_actions and b_actions
                        b_actions += c_actions
                        merged.append(c_actions)
                        inserted = True
                        break
                if inserted:
                    break
            if inserted:
                break

    for item in merged:
        ordered_idx.remove(item)

    ordered_list = []
    for o_idx in range(len(ordered_idx)):
        curr_actions = ordered_idx[o_idx]
        if len(curr_actions) == 1:
            ordered_list.append([idx2actions[curr_actions[0]]])
        else:
            new_list = []
            for curr_idx in curr_actions:
                new_list.append(idx2actions[curr_idx])
            ordered_list.append(new_list)

    return ordered_list

def merge_sets(A):
    while True:
        intersected = False
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                if len(A[i].intersection(A[j])) > 0:
                    A[i] = A[i].union(A[j])
                    A.pop(j)
                    intersected = True
                    break
            if intersected:
                break
        if not intersected:
            break
    return A

def get_key_actions_topK(seqs, K):
    count_actions_dict = count_action(seqs)
    num_seq = len(seqs)
    seqs = np.concatenate([np.unique(np.array(seq)) for seq in seqs], axis=0).tolist()
    count_dict = dict(Counter(seqs))
    key_actions = [k for k, v in count_dict.items() if v == num_seq]
    if len(key_actions) > K:
        k_dict = {}
        for key in key_actions:
            k_dict[key] = count_actions_dict[key]
        k_dict = {k:v for k, v in sorted(k_dict.items(), key=lambda item:item[1], reverse=True)}
        k_list = list(k_dict.keys())[:K]
        return k_list
    else:
        return key_actions

def count_action(seqs):
    for idx in range(len(seqs)):
        if idx == 0:
            counter = Counter(seqs[idx])
        else:
            counter.update(seqs[idx])

    return counter

def get_key_seqs(seqs, key_actions):
    key_seqs = []
    for seq in seqs:
        key_seq = []
        for token in seq:
            if token in key_actions:
                key_seq.append(token)
        key_seqs.append(key_seq)
    return key_seqs

def make_key_rule(seqs, key_actions, tdmap_ori=None, actions2idx_ori=None, idx=0, with_seq=True, args=None):
    all_actions = np.concatenate([np.array(seq) for seq in seqs])
    actions = np.unique(all_actions).tolist()
    actions2idx = {a: i for i, a in enumerate(actions)}
    tdmap = get_tdmap(seqs, actions2idx)

    max_len = max([len(seq) for seq in seqs]) + 2

    # key seqs
    key_seqs = get_key_seqs(seqs, key_actions)
    key_ordered_list = get_temporal_dependency(key_seqs, tdmap, actions2idx, np.array(key_actions), args)

    seqs = [split_seq_by_action(seq, key_ordered_list) for seq in seqs]
    num_rec = [len(seq) - 1 for seq in seqs]

    # calculating the escape probability
    prob_eps = 1 / (1 + (sum(num_rec) / len(num_rec)))
    head = 'K'
    body_list = []
    body = ""
    rules = []

    seqs = sum(seqs, [])
    key_order_in_seq = to_key_index(remain_key_actions(seqs, key_actions), key_ordered_list)
    cnt = 0
    pos_seqs = ['']

    for i, key in enumerate(key_ordered_list):
        if len(key) > 1:
            key_number = list(range(i, i + len(key)))
            key_order_in_seq_ = []
            for k_order in key_order_in_seq:
                key_order_in_seq_.append(tuple([k for k in k_order if k in key_number]))
            key_head = state(head, i + 1)
            body += key_head + " "
            subkey_heads = [state(key_head, j + 1) for j in range(len(key))]
            perms = list(permutations(list(range(len(key)))))
            key_body = []
            mid_head = state('M', i + 1)
            for ish, subkey_head in enumerate(subkey_heads):
                subkey_body = ["'" + key[ish] + "'", ""]
                subkey_prob = [1.0, 0.0]
                rules.append(SingleRule(subkey_head, subkey_body, subkey_prob))

            num_perms = []
            for ip, perm in enumerate(perms):
                key_body_ = ""
                for ik, k in enumerate(perm):
                    mid_head_i = state(state(mid_head, ik + 1), ip + 1)
                    key_body_ += subkey_heads[k] + " " + mid_head_i + " "
                key_body.append(key_body_)
                num_perms.append(sum([1 if ks == perm else 0 for ks in key_order_in_seq_]))
            if sum(num_perms) == 0: pdb.set_trace()
            key_prob = [num_perm / sum(num_perms) for num_perm in num_perms]
            rules.append(SingleRule(key_head, key_body, key_prob))

            mid_seqs = [[[] for _ in subkey_heads] for _ in perms]
            mid_pos_seqs = [[[] for _ in subkey_heads] for _ in perms]
            seqs_ = []
            for seq, order in zip(seqs, key_order_in_seq_):
                option = perms.index(order)
                key_indices = [lindex(seq, k) for k in key]
                key_indices.sort()
                if i != len(key_ordered_list) - 1:
                    next_key_indexes = min([lindex(seq[key_indices[-1]:], next_key) for next_key in key_ordered_list[i + 1]]) + key_indices[-1]
                    seqs_.append(seq[next_key_indexes:])
                else:
                    seqs_.append(seq)

                for ik in range(len(key_indices)):
                    start = key_indices[ik]
                    if ik != len(key_indices) - 1:
                        mid_seqs[option][ik].append(seq[start+1:key_indices[ik+1]])
                    else:
                        if i != len(key_ordered_list) - 1:
                            mid_seqs[option][ik].append(seq[start+1:next_key_indexes])
                        else:
                            mid_seqs[option][ik].append(seq[start+1:])

            for ip in range(len(mid_seqs)):
                for ik in range(len(mid_seqs[ip])):
                    mid_seq = mid_seqs[ip][ik]
                    if with_seq:
                        mid_rules, mid_pos_seq = make_recursive_rule(mid_seq, state(mid_head, ik+1), ip+1, with_seq=with_seq)
                        mid_pos_seqs[ip][ik] = mid_pos_seq
                    else:
                        mid_rules = make_recursive_rule(mid_seq, tdmap, actions2idx, state(mid_head, ik+1), ip+1, with_seq=with_seq, args=args)
                    rules += mid_rules
            if with_seq:
                for ip, perm in enumerate(perms):
                    pos_seqs_ = ['']
                    for ik, k in enumerate(perm):
                        pos_seqs_ = concat_seqs_to_seqs(pos_seqs_, [key[k], ''], max_len)
                        pos_seqs_ = concat_seqs_to_seqs(pos_seqs_, mid_pos_seqs[ip][ik], max_len)
                    pos_seqs += pos_seqs_
        else:
            key = key[0]
            key_head = state(head, i + 1)
            body += key_head + " "
            key_body = ["'" + key + "'", ""]
            prob = [1.0, 0.0]
            rules.append(SingleRule(key_head, key_body, prob))
            if i != len(key_ordered_list) - 1:
                key_next = key_ordered_list[i + 1][0]
                mid_seqs = [seq[lindex(seq, key)+1:lindex(seq, key_next)] for seq in seqs]
                seqs = [seq[lindex(seq, key_next):] for seq in seqs]
            else:
                mid_seqs = [seq[lindex(seq, key)+1:] for seq in seqs]
            body += state('M', i + 1) + " "
            if with_seq:
                mid_rules, mid_pos_seqs = make_recursive_rule(mid_seqs, 'M', i + 1, with_seq=with_seq)
            else:
                mid_rules = make_recursive_rule(mid_seqs, tdmap, actions2idx, 'M', i + 1, with_seq=with_seq, args=args)
            rules += mid_rules

            if with_seq:
                pos_seqs = concat_seqs_to_seqs(pos_seqs, concat_seqs_to_seqs([key, ''], mid_pos_seqs, max_len), max_len)
            # print(pos_seqs)

    rules.append(SingleRule(head, [body, body + 'K'], [prob_eps, 1 - prob_eps]))

    if with_seq:
        pos_seqs_ = pos_seqs
        if max(num_rec) != 0:
            print(max(num_rec) + 2)
            input()
            for _ in range(max(num_rec) + 2):
                pos_seqs_ = list(set([' '.join(seq.lstrip().rstrip().split()) for seq in concat_seqs_to_seqs(pos_seqs_, pos_seqs, max_len)]))


    if with_seq:
        return rules, pos_seqs_
    else:
        return rules

def make_recursive_rule(seqs, tdmap_ori=None, actions2idx_ori=None, mid_head='E', idx=0, with_seq=True, args=None):
    if all([len(seq) == 0 for seq in seqs]):
        return [SingleRule(state(mid_head, idx), [''], [1.0])]
    all_actions = np.concatenate([np.array(seq) for seq in seqs])
    actions = np.unique(all_actions)
    if len(all_actions) == 0:
        if with_seq:
            return [SingleRule(state(mid_head, idx), [''], [1.0])], ['']
        else:
            return [SingleRule(state(mid_head, idx), [''], [1.0])]

    num_actions = dict(Counter(all_actions))
    actions2idx = {a: i for i, a in enumerate(actions)}
    tdmap=None

    ordered_list = get_temporal_dependency(seqs, tdmap, actions2idx, actions, args)

    head = state(mid_head, idx)
    possible_seqs = ['']
    max_len = max([len(seq) for seq in seqs]) + 2

    if len(ordered_list) == 1 and len(ordered_list[0]) == 1:
        body = ["'" + ordered_list[0][0] + "'", '']
        p = num_actions[ordered_list[0][0]] / len(seqs)
        prob = [p, (1-p)]

        rules = [SingleRule(head, body, prob)]
        if with_seq:
            possible_seqs.append(ordered_list[0][0])
    else:
        rules = []
        body = ""
        for j, ordered_elem in enumerate(ordered_list):
            sub_head = state(head, chr(j + 97))
            body += sub_head + ' '
            if len(ordered_elem) == 1:
                sub_body_list = ["'" + ordered_elem[0] + "'", '']
                p = num_actions[ordered_elem[0]] / len(seqs)
                prob_list = [p, (1-p)]

                rule = SingleRule(sub_head, sub_body_list, prob_list)
                rules.append(rule)
                if with_seq:
                    possible_seqs = list(set([' '.join(seq.lstrip().rstrip().split()) for seq in concat_seqs_to_seqs(possible_seqs, [ordered_elem[0], ''], max_len)]))
            else:
                sub_body_list = []
                sub_prob_list = []
                num_recs = []
                all_body = []
                all_body_from_second_ = []
                for seq in seqs:
                    num_actions_in_seq = Counter(seq)
                    num_bodys_in_seq = sum([num_actions_in_seq[ordered_elem[k]] for k in range(len(ordered_elem))])
                    if num_bodys_in_seq == 0:
                        continue
                    else:
                        num_recs.append(num_bodys_in_seq - 1)
                    all_body += seq
                    all_body_from_second_ += (seq[1:] if len(seq) != 0 else [])

                all_body = dict(Counter(all_body))
                all_body_from_second = defaultdict(int)
                all_body_from_second.update(dict(Counter(all_body_from_second_)))
                num_all_body = sum([num_actions[ordered_elem[k]] for k in range(len(ordered_elem))])
                num_rec = sum(num_recs) / len(num_recs)
                if num_rec < 0: num_rec = 0
                p_eps_rec = 1 / (1 + num_rec)
                for k in range(len(ordered_elem)):
                    subsub_head = state(sub_head, chr(k + 97))
                    sub_body = "'" + ordered_elem[k] + "' " + subsub_head
                    sub_prob = sum([1 if len(seq) != 0 and ordered_elem[k] == seq[0] else 0 for seq in seqs]) / len(seqs)
                    sub_body_list.append(sub_body)
                    sub_prob_list.append(sub_prob)

                    subsub_body_list = []
                    subsub_prob_list = []
                    wo_subsub_self = ordered_elem.copy()
                    wo_subsub_self.remove(ordered_elem[k])
                    num_list = list(range(len(ordered_elem)))
                    wo_subsub_self_num = num_list.copy()
                    wo_subsub_self_num.remove(k)

                    for h in range(len(wo_subsub_self)):
                        subsub_body = "'" + wo_subsub_self[h] + "' " + state(sub_head, chr(wo_subsub_self_num[h] + 97))

                        if p_eps_rec == 1:
                            subsub_prob = 0
                        else:
                            if (sum([all_body_from_second[k] for k in ordered_elem]) - all_body_from_second[ordered_elem[k]]) == 0:
                                subsub_prob = 0
                            else:
                                subsub_prob = (1 - p_eps_rec) * all_body_from_second[wo_subsub_self[h]] / (sum([all_body_from_second[k] for k in ordered_elem]) - all_body_from_second[ordered_elem[k]])
                        subsub_body_list.append(subsub_body)
                        subsub_prob_list.append(subsub_prob)
                    subsub_body_list.append('')
                    subsub_prob_list.append(p_eps_rec)
                    if sum(subsub_prob_list) < 1.0:
                        subsub_prob_list = [subsub_prob / sum(subsub_prob_list) for subsub_prob in subsub_prob_list]
                    subsub_rule = SingleRule(subsub_head, subsub_body_list, subsub_prob_list)
                    rules.append(subsub_rule)

                sub_body_list.append('')
                sub_prob_list.append(1 - sum(sub_prob_list))
                rule = SingleRule(sub_head, sub_body_list, sub_prob_list)
                rules.append(rule)

            seqs = [list(filter(lambda a: a not in ordered_elem, seq)) for seq in seqs]
        rules.append(SingleRule(head, [body], [1.0]))
    if not with_seq:
        return rules
    return rules, possible_seqs


def get_grammar(seqs, args, with_seq=True):
    rules = []
    actions = np.unique(np.concatenate([np.array(seq) for seq in seqs]))
    actions2idx = {a: i for i, a in enumerate(actions)}

    tdmap = get_tdmap(seqs, actions2idx)

    if args.topK:
        K = args.K
        key_actions = get_key_actions_topK(seqs, K)
    else:
        key_actions = get_key_actions(seqs)
    fseqs, lseqs, mseqs = get_first_last_actions(seqs, key_actions, actions, actions2idx)

    if args.dataset == 'breakfast':
        rules.append(SingleRule('S', ["'SIL' " + state('E', 0) + " K " + state('E', 1) + ' ' + "'SIL'"],
                                [1.0]))
    elif args.dataset == '50salads':
        rules.append(SingleRule('S', ["'action_start' " + state('E', 0) + " K " + state('E', 1) + ' ' + "'action_end'"],
                                [1.0]))
    else:
        rules.append(SingleRule('S', [state('E', 0) + " K " + state('E', 1)],
                                [1.0]))
    if with_seq:
        first_rule, first_seqs = make_recursive_rule(fseqs, idx=0, with_seq=with_seq)
        last_rule, last_seqs = make_recursive_rule(lseqs, idx=1, with_seq=with_seq)
        key_rule, key_seqs = make_key_rule(mseqs, key_actions, 2, with_seq=with_seq)
    else:
        first_rule = make_recursive_rule(fseqs, tdmap, actions2idx, idx=0, with_seq=with_seq, args=args)
        last_rule = make_recursive_rule(lseqs, tdmap, actions2idx, idx=1, with_seq=with_seq, args=args)

        key_rule = make_key_rule(mseqs, key_actions, tdmap, actions2idx, 2, with_seq=with_seq, args=args)


    rules = rules + first_rule + last_rule + key_rule
    max_len = max([len(seq) for seq in seqs]) + 2

    if with_seq:
        all_pos_seqs = concat_seqs_to_seqs(concat_seqs_to_seqs(first_seqs, key_seqs, max_len), last_seqs, max_len)
        all_pos_seqs = list(set([' '.join(all_pos_seq.lstrip().rstrip().split()) for all_pos_seq in all_pos_seqs]))

    rule_wo_zero = []
    for rule in rules:
        rule.prob_list = [round(prob, 4) for prob in rule.prob_list]

    new_rules = []
    for rule in rules:
        body_list_ = []
        prob_list_ = []

        for body, prob in zip(rule.body_list, rule.prob_list):
            if prob != 0.0:
                body_list_.append(body)
                prob_list_.append(prob)

        rule.body_list = body_list_
        rule.prob_list = prob_list_

    if not with_seq:
        return rules

    return rules, all_pos_seqs

def merge_rules(rules, pre_str):
    for rule in rules:
        body_list = []
        prob_list = []
        rule.head = pre_str+rule.head
        for body, prob in zip(rule.body_list, rule.prob_list):
            bodies = body.split()
            new_body = ''
            for b in bodies:
                if b[0] == 'E' or b[0] == 'K' or b[0] == 'M':
                    new_body += pre_str+b
                else:
                    new_body += b
                new_body += ' '
            new_body = new_body[:-1]
            body_list.append(new_body)
        rule.body_list = body_list

    return rules

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='induced_grammars', help='results directory to save the induced grammars')
    parser.add_argument('--source_dir', type=str, default='source_breakfast', help='source directory of action sequences')
    parser.add_argument('--dataset', type=str, default='breakfast', help='choose from [breakfast/50salads]')
    parser.add_argument('--seq', default=False, action='store_true', help='extract the sequences simultaneously')
    parser.add_argument('--topK', action='store_true', help='maintain the Top-K number of key actions')
    parser.add_argument('--K', type=int, default=3, help='hyperparameter for K')
    parser.add_argument('--merge', action='store_true', help='merge all activity grammars for Breakfast')
    args = parser.parse_args()

    mapping_file = args.source_dir+'/mapping.txt'
    file_ptr = open(mapping_file, 'r')
    if args.dataset == 'breakfast':
        activities = open(os.path.join(args.source_dir, 'activity_category.txt'), 'r').readlines()
    else:
        activities = [args.dataset]
    actions = file_ptr.read().split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    if args.dataset == '50salads':
        splits = [1, 2, 3, 4, 5]
    else:
        splits = [1, 2, 3, 4]
    for split in splits:
        source_dir = os.path.join( args.source_dir, 'split'+str(split))
        save_dir = os.path.join('results', args.result_dir, args.dataset, 'split'+str(split))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if args.merge:
            merged_grammar = []
            root_rule = 'S -> '
        idx=0
        for activity in activities:
            activity = activity.split(' ')[-1].split('\n')[0]
            if args.dataset == 'breakfast':
                source_file = os.path.join(source_dir, activity + '.txt')
            else:
                source_file = source_dir+'.txt'
            seqs = open(source_file, 'r').readlines()
            # exclude SIL (breakfast) / action_start & action_end (50salads)
            seqs = [seq.split(' ')[2:-2] for seq in seqs]

            grammar = get_grammar(seqs, args, with_seq=False)

            #merge all activity grammars for Breakfast
            if args.merge:
                if idx == 4 or idx == 10:
                    idx +=1
                pre_str = chr(65+idx)
                merged_grammar.append(SingleRule('S', [pre_str+'S'], [1.0/len(activities)]))
                for rule in merge_rules(grammar, pre_str):
                    merged_grammar.append(rule)
            else:
                w_path = os.path.join(save_dir, activity+'.pcfg')
                with open(w_path, 'w') as w:
                    for rule in grammar:
                        w.write(rule.__repr__() + '\n')
            idx+=1
        if args.merge:
            w_path = os.path.join(save_dir, 'all.pcfg')
            with open(w_path, 'w') as w:
                for rule in merged_grammar:
                    w.write(rule.__repr__() + '\n')
    print('Grammar induction complete!')



if __name__ == '__main__':
    main()
