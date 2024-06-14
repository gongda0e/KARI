"""
This code is an implementation of the Breadth-first Earley Parser, introduced in our paper:
Activity Grammars for Temporal Action Segmentation, NeurIPS 2023 submission

Created on May 17, 2023

This code is based on Generalized Earley Parser by Siyuan Qi, available at https://github.com/Buzz-Beater/GEP_PAMI/tree/master.
"""

import heapq

import numpy as np
import grammarutils as grammarutils
import nltk.grammar
from tqdm import tqdm
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import copy

def format_num(num):
    if num > 1e-3 or num == 0:
        return '{:.3f}'.format(num)
    else:
        return '{:.1e}'.format(num)


class State(object):
    def __init__(self, r, dot, start, end, i, j, k, rule_index, operation, last_i, last_j, last_rule_index, prefix, prob, forward, inner):
        self._r = r
        self._dot = dot
        self._start = start
        self._end = end

        # The parent rule is indexd by state_set[i][j][rule_index]
        self._i = i
        self._j = j
        self._k = k
        self._rule_index = rule_index
        self._operation = operation

        self._last_i = last_i
        self._last_j = last_j
        self._last_rule_index = last_rule_index

        self._prefix = prefix
        self._prob = prob

        # Stockle, A. An Efficient Probabilistic Context-Free Parsing Algorithm that Computes Prefix Probabilities. (1995)
        # https://www.aclweb.org/anthology/J95-2002
        # In probability range
        self._forward = forward
        self._inner = inner

    def is_complete(self):
        return self._dot == len(self._r.rhs())

    def next_symbol(self):
        if self.is_complete():
            return None
        return self._r.rhs()[self._dot]

    def earley_equivalent(self, other_state):
        return self.earley_hash() == other_state.earley_hash()

    def earley_hash(self):
        rhs = [str(n) for n in self._r.rhs()]
        return '[{}:{}:{}] {} -> {}: {}'.format(self._dot, self._start, self._end, self._r.lhs(), rhs, self.prefix_str())

    def prefix_str(self):
        return ' '.join(self._prefix)

    def __repr__(self):
        rhs = [str(n) for n in self._r.rhs()]
        rhs = ' '.join(rhs[:self._dot]) + ' * ' + ' '.join(rhs[self._dot:])
        return '{} -> {} : {:.3f} ``{}" (start: {} end: {}) (forward:{}, inner:{}, {} from [{}:{}:{}:{}])'\
            .format(self._r.lhs(), rhs, self._prob, ' '.join(self._prefix), self.start, self.end, self._forward, self._inner,
                    self._operation, self._i, self._j, self._k, self._rule_index)

    def tex(self, state_idx, prefix_tex, state_sets):
        if str(self._r.lhs()) == 'GAMMA':
            lhs = '\\Gamma'
        else:
            lhs = str(self._r.lhs())
        rhs = [str(n) for n in self._r.rhs()]
        rhs = ' '.join(rhs[:self._dot]) + ' \\boldsymbol{\\cdot} ' + ' '.join(rhs[self._dot:])
        rule = lhs + ' \\rightarrow ' + rhs

        if self._operation == 'root':
            comment = 'start rule'
        elif self._operation == 'predict':
            comment = 'predict: ({})'.format(self._rule_index)
        elif self._operation == 'scan':
            comment = 'scan: S({}, {})({})'.format(self._last_i, self._last_j, self._last_rule_index)
        elif self._operation == 'complete':
            comment = 'complete: ({}) and S({}, {})({})'.format(self._last_rule_index[1], self._last_i, self._last_j, self._last_rule_index[0])

        return '({}) & ${}$ & {} & {} & ``${}$" & {}\\\\'.format(state_idx, rule, format_num(self._forward), format_num(self._inner), prefix_tex, comment)

    @property
    def r(self): return self._r

    @property
    def dot(self): return self._dot

    @property
    def start(self): return self._start

    @property
    def end(self): return self._end

    @property
    def i(self): return self._i

    @property
    def j(self): return self._j

    @property
    def k(self): return self._k

    @property
    def rule_index(self): return self._rule_index

    @property
    def prefix(self): return self._prefix

    @property
    def prob(self): return self._prob

    @property
    def forward(self): return self._forward

    @property
    def inner(self): return self._inner

    @prob.setter
    def prob(self, value):
        self._prob = value

    @forward.setter
    def forward(self, value):
        self._forward = value

    @inner.setter
    def inner(self, value):
        self._inner = value

class BreadthFirstEarley(object):
    def __init__(self, grammar, prior_flag=False, mapping=None, priority= 'd', prune_prob = False):
        self._grammar = grammar
        self._classifier_output = None
        self._total_frame = 0
        self._cached_log_prob = None
        self._cached_grammar_prob = None
        self._state_set = None
        self._queue = None
        self._prefix_queue = None
        self._max_log_prob = None
        self._best_l = None
        self._mapping = mapping
        self.priority = priority
        self._parse_init()
        self.prior = prior_flag
        self.parsed_str = []
        self.prune_prob = prune_prob

    def _parse_init(self, classifier_output=None):
        self._queue = []
        self._state_set = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))                #####STATE_SET: [M][N][D] = list
        for r in self._grammar.productions():
            if str(r.lhs()) == 'GAMMA':
                self._state_set[0][0][0].append(State(r, 0, 0, 0, 0, 0, 0, -1, 'root', 0, 0, 0, [], 0.0, 1.0, 1.0))
                break
        if self.priority == 'd':
            heapq.heappush(self._queue, ((0, (-np.inf, 0, 0, '', self._state_set[0][0][0]))))               ############SET DEPTH AS PRIORITY#############
        elif self.priority == 'dp':
            heapq.heappush(self._queue, (((0, -np.inf), (-np.inf, 0, 0, '', self._state_set[0][0][0]))))               ############SET DEPTH AS PRIORITY#############
        elif self.priority == 'pd':
            heapq.heappush(self._queue, (((-np.inf, 0), (-np.inf, 0, 0, '', self._state_set[0][0][0]))))               ############SET DEPTH AS PRIORITY#############
        elif self.priority == 'lpd':
            heapq.heappush(self._queue, (((0, -np.inf, 0), (-np.inf, 0, 0, '', self._state_set[0][0][0]))))               ############SET DEPTH AS PRIORITY#############

        self._max_log_prob = -np.inf

        if classifier_output is not None:
            if len(classifier_output.shape) != 2:
                raise ValueError('Classifier output shape not recognized, expecting (frame_num, class_num).')
            self._classifier_output = classifier_output
            self._cached_log_prob = dict()
            self._cached_grammar_prob = dict()
            self._total_frame = self._classifier_output.shape[0]
            self._class_num = self._classifier_output.shape[1]
            self._cached_log_prob[''] = np.ones(self._total_frame + 1) * np.finfo('d').min
            self._cached_log_prob[''][self._total_frame] = 0.0
            self._cached_grammar_prob[''] = 1.0

    def state_set_vis(self):
        for m, m_set in self._state_set.items():
            print('======================================================')
            for n, mn_set in m_set.items():
                for d, mnd_set in mn_set.items():
                    prefix_str = mnd_set[0].prefix_str()
                    for state_idx, state in enumerate(mnd_set):
                        print('[{} {} {} / {}] {}, prior: {}, prefix: {}, parsing: {}'.format(
                                                            m, n, d, state_idx, state, self._cached_grammar_prob[prefix_str],
                                                            np.exp(self._cached_log_prob[prefix_str][self._total_frame]),
                                                            np.exp(self._cached_log_prob[prefix_str][self._total_frame-1])
                                                        ))
            print('======================================================')

    def state_set_tex(self):
        print('\\begin{tabular}{|c|l|l|l|l|l|}\n\\hline\nstate \\# & rule & $\\mu$ & $\\nu$ & prefix & comment \\\\\n\\hline')
        for m, m_set in self._state_set.items():
            for n, mn_set in m_set.items():
                for d, mnd_set in mn_set.items():
                    prefix_str = mnd_set[0].prefix_str()
                    prefix_tex = prefix_str or '\\epsilon'

                    grammar_prior = format_num(self._cached_grammar_prob[prefix_str])
                    parsing_prob = format_num(np.exp(self._cached_log_prob[prefix_str][self._total_frame-1]))
                    prefix_prob = format_num(np.exp(self._cached_log_prob[prefix_str][self._total_frame]))
                    print('\\multicolumn{{6}}{{l}}{{$S({}, {}): l=``{}", p(l|G)={}, p(l|x, G)={}, p(l_{{\\cdots}}|x, G)={}$}} \\\\'
                          .format(m, n, prefix_tex, grammar_prior, parsing_prob, prefix_prob))
                    print('\\hline')
                    for state_idx, state in enumerate(mn_set):
                        print(state.tex(state_idx, prefix_tex, self._state_set))
                    print('\\hline')
        print('\\multicolumn{5}{l}{Final output: $l^{*} = ``0 + 1"$ with probability 0.054} \\\\\n\\end{tabular}')

    def cached_prob_tex(self):
        print('\\begin{{tabular}}{{|{}|}}'.format('|'.join(['c']*(len(self._cached_log_prob.keys())+1))))
        print('\\hline')
        prefices = list(self._cached_log_prob.keys())[:]
        prefices.sort(key=lambda item: (len(item), item))
        print('Frame & $\\epsilon$' + ' & '.join(prefices) + ' \\\\')
        print('\\hline')
        for f in range(self._total_frame+1):
            print('{} & '.format(f) + ' & '.join([format_num(np.exp(self._cached_log_prob[prefix][f])) for prefix in prefices]) + ' \\\\')
        print('\\hline')
        print('\\end{tabular}')

    def debug(self, verbose=False):
        if verbose:
            for l, p in self._cached_log_prob.items():
                print(l, p)

    def parse(self, classifier_output, str_len=20, prune=20):
        start = time.time()

        # initialize parser
        self._parse_init(classifier_output)
        count = 0
        while self._queue:
            count += 1
            #print(self._queue)
            # Remove queue elements exceeding the size
            if self.prune_prob:
                q_p = copy.deepcopy(self._queue)
                q_p=sorted(q_p, key=lambda x: x[1][0])
                if prune >0:
                    if len(q_p)>prune:
                        del q_p[prune:]
                q_d = copy.deepcopy(q_p)
                q_d=sorted(q_d, key=lambda x: x[0])
                self._queue = copy.deepcopy(q_d)
            else:
                if prune > 0:
                    if len(self._queue) > prune:
                        del self._queue[prune:]

            if self.priority == 'd':
                d, (pp, m, n, set_l, current_set) = heapq.heappop(self._queue)
            elif self.priority == 'dp':
                (d,pp), (pp, m, n, set_l, current_set) = heapq.heappop(self._queue)
            elif self.priority == 'pd':
                (pp,d), (pp, m, n, set_l, current_set) = heapq.heappop(self._queue)
            elif self.priority == 'lpd':
                (_, pp,d), (pp, m, n, set_l, current_set) = heapq.heappop(self._queue)

            branch_log_probs = dict()
            # branch_log_probs[set_l] previously have the prefix probability of "set_l" string
            # since we are expanding this path on the prefix tree, we update this probability to parsing probability
            branch_log_probs[set_l] = self._cached_log_prob[set_l][self._total_frame - 1]
            if self._cached_log_prob[set_l][self._total_frame - 1] > self._max_log_prob:
                self._max_log_prob = self._cached_log_prob[set_l][self._total_frame - 1]
                self._best_l = set_l

            # get updating states
            new_completed_states = list()
            new_predicted_states = list()
            new_scanned_states = list()
            for rule_index, s in enumerate(current_set):
                if s.is_complete():
                    new_completed_states.append(self.complete(m, n, d, rule_index, s))
                elif nltk.grammar.is_nonterminal(s.next_symbol()):
                    new_predicted_states.append(self.predict(m, n, d, rule_index, s))
                elif nltk.grammar.is_terminal(s.next_symbol()):
                    if m == self._total_frame:
                        continue
                    new_scanned_states.append(self.scan(m, n, d, rule_index, s))
                else:
                    raise ValueError('No operation (predict, scan, complete) applies to state {}'.format(s))

            # Push states from completion
            if sum(new_completed_states)!=0 and (d-1)!=0:
                if self.priority == 'd':
                    heapq.heappush(self._queue, (d-1, (pp, m, n, set_l, self._state_set[m][n][d-1])))
                elif self.priority == 'dp':
                    heapq.heappush(self._queue, ((d-1, pp), (pp, m, n, set_l, self._state_set[m][n][d-1])))
                elif self.priority == 'pd':
                    heapq.heappush(self._queue, ((pp, d-1), (pp, m, n, set_l, self._state_set[m][n][d-1])))
                elif self.priority == 'lpd':
                    heapq.heappush(self._queue, ((len(set_l.split()), pp, d-1), (pp, m, n, set_l, self._state_set[m][n][d-1])))

            # Check if parsing has finished
            if sum(new_completed_states)!=0 and d==1:
                if set_l not in self.parsed_str:
                    self.parsed_str.append(set_l)

            # Push states from prediction
            if sum(new_predicted_states)!=0:
                if self.priority == 'd':
                    try:
                        heapq.heappush(self._queue, (d+1, (pp, m, n, set_l, self._state_set[m][n][d+1])))
                    except TypeError:
                        pass
                elif self.priority == 'dp':
                    heapq.heappush(self._queue, ((d+1,pp), (pp, m, n, set_l, self._state_set[m][n][d+1])))
                elif self.priority == 'pd':
                    heapq.heappush(self._queue, ((pp, d+1), (pp, m, n, set_l, self._state_set[m][n][d+1])))
                elif self.priority == 'lpd':
                    heapq.heappush(self._queue, ((len(set_l.split()), pp, d+1), (pp, m, n, set_l, self._state_set[m][n][d+1])))

            # Push states from scanning
            for new_m, new_n, new_d, new_prefix_str in new_scanned_states:
                prefix_len = len(new_prefix_str.split(' '))
                new_prefix = self._state_set[new_m][new_n][new_d][0].prefix

                if prefix_len > str_len:         #Skip pushing the state into the queue since it already parsed more than "str_len" actions
                    continue

                if not new_prefix_str in self._cached_grammar_prob.keys():
                    self._cached_grammar_prob[new_prefix_str] = 0
                    for new_s in self._state_set[new_m][new_n][new_d]:
                        self._cached_grammar_prob[new_prefix_str] += new_s.forward

                    prob = self.compute_prob(new_prefix, None)
                    for new_s in self._state_set[new_m][new_n][new_d]:
                        new_s.prob = prob
                    if self.priority == 'd':
                        heapq.heappush(self._queue, (new_d, (-prob, new_m, new_n, new_prefix_str, self._state_set[new_m][new_n][new_d])))
                    elif self.priority == 'dp':
                        heapq.heappush(self._queue, ((new_d, -prob), (-prob, new_m, new_n, new_prefix_str, self._state_set[new_m][new_n][new_d])))
                    elif self.priority == 'pd':
                        heapq.heappush(self._queue, ((-prob, new_d), (-prob, new_m, new_n, new_prefix_str, self._state_set[new_m][new_n][new_d])))
                    elif self.priority == 'lpd':
                        heapq.heappush(self._queue, ((len(new_prefix_str.split()), -prob, new_d), (-prob, new_m, new_n, new_prefix_str, self._state_set[new_m][new_n][new_d])))
                else:
                    tmp_cached_grammar_prob = 0
                    for new_s in self._state_set[new_m][new_n][new_d]:
                        tmp_cached_grammar_prob += new_s.forward
                    prob = self.compute_prob(new_prefix, tmp_cached_grammar_prob)
                    for new_s in self._state_set[new_m][new_n][new_d]:
                        new_s.prob = prob
                    if self.priority == 'd':
                        heapq.heappush(self._queue, (new_d, (-prob, new_m, new_n, new_prefix_str, self._state_set[new_m][new_n][new_d])))
                    elif self.priority == 'dp':
                        heapq.heappush(self._queue, ((new_d,-prob), (-prob, new_m, new_n, new_prefix_str, self._state_set[new_m][new_n][new_d])))
                    elif self.priority == 'pd':
                        heapq.heappush(self._queue, ((-prob, new_d), (-prob, new_m, new_n, new_prefix_str, self._state_set[new_m][new_n][new_d])))
                    elif self.priority == 'lpd':
                        heapq.heappush(self._queue, ((len(new_prefix_str), -prob, new_d), (-prob, new_m, new_n, new_prefix_str, self._state_set[new_m][new_n][new_d])))

            # Early stop
            if self._queue:
                max_prefix_log_prob = -np.inf
                for node in self._queue:
                    pfx = node[1][3]
                    if self._cached_log_prob[pfx][self._total_frame] > max_prefix_log_prob:
                        max_prefix_log_prob = self._cached_log_prob[pfx][self._total_frame]
            else:
                max_prefix_log_prob = -np.inf

            if len(self.parsed_str)>=1:
                str_prob_dict = {}
                for key in set(self.parsed_str):
                    str_prob_dict[key] = self._cached_log_prob[key][self._total_frame-1]
                str_prob_dict = {k:v for k, v in sorted(str_prob_dict.items(), key=lambda item: item[1])}
                max_parsed_log_prob = max(str_prob_dict.values())
                if max_parsed_log_prob > max_prefix_log_prob:
                    self._best_l = max(str_prob_dict, key=str_prob_dict.get)
                    self._max_log_prob = max_parsed_log_prob
                    return self._best_l, self._max_log_prob, self._cached_log_prob

        # Only sequences which finished parsing can be returned.
        self.debug()
        if len(self.parsed_str) != 0:
            str_prob_dict = {}
            for key in self.parsed_str:
                str_prob_dict[key] = self._cached_log_prob[key][self._total_frame-1]
            if self._best_l not in str_prob_dict.keys():
                self._best_l = max(str_prob_dict, key=str_prob_dict.get)
                self._max_log_prob = max(str_prob_dict.values())
            return self._best_l, self._max_log_prob, self._cached_log_prob
        else:
            return self._best_l, self._max_log_prob, self._cached_log_prob

    def get_log_prob_sum(self):
        log_prob = np.log(self._classifier_output).transpose()
        log_prob_sum = np.zeros((self._class_num, self._total_frame, self._total_frame))
        for c in range(self._class_num):
            for b in range(self._total_frame):
                log_prob_sum[c, b, b] = log_prob[c, b]
        for c in range(self._class_num):
            for b in range(self._total_frame):
                for e in range(b+1, self._total_frame):
                    log_prob_sum[c, b, e] = log_prob_sum[c, b, e-1] + log_prob[c, e]
        return log_prob, log_prob_sum

    def compute_labels(self):
        log_prob, log_prob_sum = self.get_log_prob_sum()

        tokens = [int(token) for token in self._best_l.split(' ')]
        dp_tables = np.zeros((len(tokens), self._total_frame))
        traces = np.zeros_like(dp_tables)

        for end in range(0, self._total_frame):
            dp_tables[0, end] = log_prob_sum[tokens[0], 0, end]

        for token_i, token in enumerate(tokens):
            if token_i == 0:
                continue
            for end in range(token_i, self._total_frame):
                max_log_prob = -np.inf
                for begin in range(token_i, end+1):
                    check_prob = dp_tables[token_i-1, begin-1] + log_prob_sum[token, begin, end]
                    if check_prob > max_log_prob:
                        max_log_prob = check_prob
                        traces[token_i, end] = begin-1
                dp_tables[token_i, end] = max_log_prob

        # Back tracing
        token_pos = [-1 for _ in tokens]
        token_pos[-1] = self._total_frame - 1
        for token_i in reversed(range(len(tokens)-1)):
            token_pos[token_i] = int(traces[token_i+1, token_pos[token_i+1]])

        labels = - np.ones(self._total_frame).astype(np.int)
        labels[:token_pos[0]+1] = tokens[0]
        for token_i in range(1, len(tokens)):
            labels[token_pos[token_i-1]+1:token_pos[token_i]+1] = tokens[token_i]

        return labels, self._best_l.split(' '), token_pos


    def compute_labels_segment(self, classifier_output):
        #####
        self._classifier_output = classifier_output
        self._total_frame = np.shape(classifier_output)[0]
        #####
        log_prob, log_prob_sum = self.get_log_prob_sum()

        tokens = [int(token) for token in self._best_l.split(' ')]
        dp_tables = np.zeros((len(tokens), self._total_frame))
        traces = np.zeros_like(dp_tables)

        for end in range(0, self._total_frame):
            dp_tables[0, end] = log_prob_sum[tokens[0], 0, end]

        for token_i, token in enumerate(tokens):
            if token_i == 0:
                continue
            for end in range(token_i, self._total_frame):
                max_log_prob = -np.inf
                for begin in range(token_i, end+1):
                    check_prob = dp_tables[token_i-1, begin-1] + log_prob_sum[token, begin, end]
                    if check_prob > max_log_prob:
                        max_log_prob = check_prob
                        traces[token_i, end] = begin-1
                dp_tables[token_i, end] = max_log_prob

        # Back tracing
        token_pos = [-1 for _ in tokens]
        token_pos[-1] = self._total_frame - 1
        for token_i in reversed(range(len(tokens)-1)):
            token_pos[token_i] = int(traces[token_i+1, token_pos[token_i+1]])

        labels = - np.ones(self._total_frame).astype(int)
        labels[:token_pos[0]+1] = tokens[0]
        for token_i in range(1, len(tokens)):
            labels[token_pos[token_i-1]+1:token_pos[token_i]+1] = tokens[token_i]

        return labels, self._best_l.split(' '), token_pos

    def complete(self, m, n, d, rule_index, s):
        comp_sw = 0
        for back_s in self._state_set[s.i][s.j][d-1]:
            if str(back_s.next_symbol()) == str(s.r.lhs()) and back_s.end == s.start:
                forward_prob = back_s.forward * s.inner
                inner_prob = back_s.inner * s.inner
                new_s = State(back_s.r, back_s.dot + 1, back_s.start, s.end, back_s.i, back_s.j, back_s.k, back_s.rule_index,
                              'complete', s.i, s.j, (s.rule_index, rule_index), s.prefix, s.prob, forward_prob, inner_prob)

                # Stockle, A. 1995 p176 completion probability calculation
                state_exist = False
                for r_idx, exist_s in enumerate(self._state_set[m][n][d-1]):
                    if exist_s.earley_equivalent(new_s):
                        assert (not state_exist), 'Complete duplication'
                        state_exist = True
                        #del self._state_set[m][n][d-1][r_idx]
                        exist_s.forward += forward_prob
                        exist_s.inner += inner_prob

                if not state_exist:
                    self._state_set[m][n][d-1].append(new_s)
                    comp_sw = 1
        return comp_sw

    def predict(self, m, n, d, rule_index, s):
        expand_symbol = str(s.next_symbol())
        pred_sw=0
        for r in self._grammar.productions():
            production_prob = r.prob()
            forward_prob = s.forward * production_prob
            inner_prob = production_prob

            if expand_symbol == str(r.lhs()):
                new_s = State(r, 0, s.end, s.end, m, n, d, rule_index, 'predict', m, n, rule_index, s.prefix, s.prob, forward_prob, inner_prob)

                # Stockle, A. 1995 p176 prediction probability calculation
                state_exist = False
                for r_idx, exist_s in enumerate(self._state_set[m][n][d+1]):
                    if exist_s.earley_equivalent(new_s):
                        assert(not state_exist), 'Prediction duplication'
                        state_exist = True
                        #del self._state_set[m][n][d+1][r_idx]
                        exist_s.forward += forward_prob

                if not state_exist:
                    self._state_set[m][n][d+1].append(new_s)
                    pred_sw=1
        return pred_sw

    def scan(self, m, n, d, rule_index, s):
        new_prefix = s.prefix[:]
        new_prefix.append(str(s.next_symbol()))

        forward_prob = s.forward
        inner_prob = s.inner

        new_s = State(s.r, s.dot + 1, s.start, s.end + 1, s.i, s.j, s.k, s.rule_index, 'scan', m, n, rule_index, new_prefix, 0.0, forward_prob, inner_prob)

        if m == len(self._state_set) - 1:
            new_n = 0
        else:
            new_n = len(self._state_set[m + 1])

        new_prefix_str = new_s.prefix_str()

        self._state_set[m + 1][new_n][d].append(new_s)
        return m + 1, new_n, d, new_prefix_str

    def compute_prob(self, prefix, tmp_grammar_prob):
        l = ' '.join(prefix)
        l_minus = ' '.join(prefix[:-1])

        # Store grammar transition probability
        # Assume l = {l_minus, k}
        # self._cached_grammar_log_prob[l] = log(p(l | G))
        # transtition_log_prob = log( p(l... | G) / p(l_minus... | G)) = log(p(k | l_minus, G))
        if self.prior:
            transition_log_prob = np.log(self._cached_grammar_prob[l]) - np.log(self._cached_grammar_prob[l_minus])
            if not tmp_grammar_prob is None:
                transition_log_prob = np.log(tmp_grammar_prob) - np.log(self._cached_grammar_prob[l_minus])
        else:
            transition_log_prob = 0


        if l not in self._cached_log_prob:
            if self._mapping:
                k = int(self._mapping[prefix[-1]])
            else:
                k = int(prefix[-1])
            l_minus = ' '.join(prefix[:-1])

            # Initialize p(l|x_{0:T}) to negative infinity
            self._cached_log_prob[l] = np.ones(self._total_frame + 1) * np.finfo('d').min
            if len(prefix) == 1:
                # Initializiation for p(l|x_{0:T}) when T = 0 and l only contains one symbol
                self._cached_log_prob[l][0] = np.log(self._classifier_output[0, k]) + transition_log_prob
                # self._cached_log_prob[l][0] = np.log(self._classifier_output[0, k])

            # Compute p(l)
            for t in range(1, self._total_frame):
                # To prevent numerical underflow for np.exp when the exponent is too small
                # In the meanwhile, prevent overflow for np.exp(maximum - regularizer)

                epsilon = 0
                max_log = max(self._cached_log_prob[l][t - 1],
                              (1-epsilon)*self._cached_log_prob[l_minus][t - 1] + transition_log_prob)

                # p(l|x_{0:t}, G) = y_t^{k} (p(l | x_{0:t-1}, G) + p(k | l^{-}, G) p(l^{-} | x_{0:t-1}, G))

                self._cached_log_prob[l][t] \
                    = np.log(self._classifier_output[t, k]) + max_log + \
                    np.log(np.exp(self._cached_log_prob[l][t - 1] - max_log) +
                    np.exp((1-epsilon)*self._cached_log_prob[l_minus][t - 1] + transition_log_prob - max_log))

            # Compute p(l...)
            if self._total_frame == 1:
                # When only 1 frame, p(l...|x_{0:t}) = p(l... | x_0) = p(l | x_0)
                self._cached_log_prob[l][self._total_frame] = self._cached_log_prob[l][0]
            else:
                max_log = max(self._cached_log_prob[l][0],
                              np.max(self._cached_log_prob[l_minus][0:self._total_frame - 1] + transition_log_prob))

                # (ICML 2018) Generalized Earley parser Equation(3)
                first_part =  np.exp(self._cached_log_prob[l][0] - max_log)

                second_part = 0
                for t in range(1, self._total_frame):
                     second_part += self._classifier_output[t, k] * \
                                        np.exp(self._cached_log_prob[l_minus][t - 1] + transition_log_prob - max_log)

                self._cached_log_prob[l][self._total_frame] = first_part + second_part
                self._cached_log_prob[l][self._total_frame] = \
                    np.log(self._cached_log_prob[l][self._total_frame]) + max_log
        else:
            if self._mapping:
                k = int(self._mapping[prefix[-1]])
            else:
                k = int(prefix[-1])
            l_minus = ' '.join(prefix[:-1])

            # Initialize p(l|x_{0:T}) to negative infinity
            tmp_cached_log_prob = np.ones(self._total_frame + 1) * np.finfo('d').min
            if len(prefix) == 1:
                # Initializiation for p(l|x_{0:T}) when T = 0 and l only contains one symbol
                tmp_cached_log_prob[0] = np.log(self._classifier_output[0, k]) + transition_log_prob

            # Compute p(l)
            for t in range(1, self._total_frame):
                # To prevent numerical underflow for np.exp when the exponent is too small
                # In the meanwhile, prevent overflow for np.exp(maximum - regularizer)

                max_log = max(tmp_cached_log_prob[t - 1],
                              self._cached_log_prob[l_minus][t - 1] + transition_log_prob)

                # p(l|x_{0:t}, G) = y_t^{k} (p(l | x_{0:t-1}, G) + p(k | l^{-}, G) p(l^{-} | x_{0:t-1}, G))

                tmp_cached_log_prob[t] \
                    = np.log(self._classifier_output[t, k]) + max_log + \
                    np.log(np.exp(tmp_cached_log_prob[t - 1] - max_log) +
                    np.exp(self._cached_log_prob[l_minus][t - 1] + transition_log_prob - max_log))

            # Compute p(l...)
            if self._total_frame == 1:
                # When only 1 frame, p(l...|x_{0:t}) = p(l... | x_0) = p(l | x_0)
                tmp_cached_log_prob[self._total_frame] = tmp_cached_log_prob[0]
            else:
                max_log = max(tmp_cached_log_prob[0],
                              np.max(self._cached_log_prob[l_minus][0:self._total_frame - 1] + transition_log_prob))

                # (ICML 2018) Generalized Earley parser Equation(3)
                first_part =  np.exp(tmp_cached_log_prob[0] - max_log)

                second_part = 0
                for t in range(1, self._total_frame):
                     second_part += self._classifier_output[t, k] * \
                                        np.exp(self._cached_log_prob[l_minus][t - 1] + transition_log_prob - max_log)

                tmp_cached_log_prob[self._total_frame] = first_part + second_part
                tmp_cached_log_prob[self._total_frame] = \
                    np.log(tmp_cached_log_prob[self._total_frame]) + max_log
            if self._cached_log_prob[l][self._total_frame] < tmp_cached_log_prob[self._total_frame]:
                self._cached_log_prob[l] = tmp_cached_log_prob
                self._cached_grammar_prob[l] = tmp_grammar_prob

        # Search according to prefix probability (Prefix probability stored in the last dimension)
        return self._cached_log_prob[l][self._total_frame]

    def my_cmp(a, b):
        # Compare based on condition 1
        if a[0] < b[0]:
            return -1
        elif a[0] > b[0]:
            return 1

        # Compare based on condition 2
        if a[1][0] < b[1][0]:
            return -1
        elif a[1][0] > b[1][0]:
            return 1

        # Compare based on condition 3
        if a[1][1] < b[1][1]:
            return -1
        elif a[1][1] > b[1][1]:
            return 1

        # Compare based on condition 4
        if a[1][2] < b[1][2]:
            return -1
        elif a[1][2] > b[1][2]:
            return 1

        # Compare based on condition 5
        if a[1][3] < a[1][3]:
            return -1
        elif a[1][3] > b[1][3]:
            return 1

        if len(a[1][3]) < len(a[1][3]):
            return -1
        elif len(a[1][3]) > len(b[1][3]):
            return 1

        # Elements are equal
        return 0


def main():
    pass


if __name__ == '__main__':
    main()
