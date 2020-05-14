import pandas
import numpy
import copy
from APIs import *
import time
import json
from multiprocessing import Pool
import multiprocessing
import sys
import time
import argparse
import os
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
import re
import string
from Model import Ranker
import torch
from nltk.metrics.distance import edit_distance

with open('data/freq_list.json') as f:
    vocab = json.load(f)

with open('data/stop_words.json') as f:
    stop_words = json.load(f)

months_a = ['january', 'february', 'march', 'april', 'may', 'june',
                 'july', 'august', 'september', 'october', 'november', 'december']
months_b = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
a2b = {a: b for a, b in zip(months_a, months_b)}
b2a = {b: a for a, b in zip(months_a, months_b)}


class ProgramNode(object):
    def __init__(self, name):
        self.name = name.strip()
        self.children = []

    def append_child(self, argument):
        self.children.append(argument)
        
    def __str__(self):
        if len(self.children) == 1:
            return '{}{{{}}}'.format(self.name, self.children[0])
        elif len(self.children) == 2:
            return '{}{{{}; {}}}'.format(self.name, self.children[0], self.children[1])
        elif len(self.children) == 3:
            return '{}{{{}; {}; {}}}'.format(self.name, self.children[0], self.children[1], self.children[2])
        else:
            raise NotImplementedError

def normalize(string):
    if string.startswith('.'):
        string = '0' + string

    return string

def tree_eq(tree1, tree2):
    if isinstance(tree1, str) and isinstance(tree2, str):
        if tree1 == tree2:
            return True
        else:
            tree1 = normalize(tree1)
            tree2 = normalize(tree2)
            
            if edit_distance(tree1, tree2) == 1 and len(tree1) >= 3:
                return True
            else:
                if " " + tree1 + " "  in " " + tree2 + " " or " " + tree2 + " "  in " " + tree1 + " ":
                    return True
                else:
                    return False

    elif isinstance(tree1, ProgramNode) and isinstance(tree2, ProgramNode):
        if tree1.name == tree2.name:
            if len(tree1.children) == len(tree2.children):
                if tree1.name in ['eq', 'and']:
                    t1 = tree_eq(tree1.children[0], tree2.children[0]) and tree_eq(tree1.children[1], tree2.children[1])
                    t2 = tree_eq(tree1.children[0], tree2.children[1]) and tree_eq(tree1.children[1], tree2.children[0])
                    return t1 or t2
                else:
                    return all([tree_eq(t1, t2) for t1, t2 in zip(tree1.children, tree2.children)])
            else:
                return False
        else:
            return False
    else:
        return False


def recursion(string):
    stack = []
    prev = ''
    if ' ; ' in string:
        arrays = string.split(' ')
    else:
        arrays = split_prog(string)
    for c in arrays:
        if c == '{':
            node = ProgramNode(prev)
            #if len(stack) == 0:
            stack.append(node)
            prev = ''
        elif c == '}':
            r = stack.pop(-1)
            r.append_child(prev)
            prev = r
        elif c == ';':
            try:
                stack[-1].append_child(prev)
                prev = ''
            except Exception:
                raise ValueError
        else:
            if prev == '':
                prev = c
            else:
                prev += " " + c
    return prev


def program_eq(string1, string2):
    string1 = string1.rstrip('=True').strip()
    string2 = string2.rstrip('=False').strip()
    head1 = recursion(string1)
    head2 = recursion(string2)
    return tree_eq(head1, head2)


class Node(object):
    def __init__(self, rows, memory_str, memory_num, header_str, header_num, must_have, must_not_have):
        # For intermediate results
        self.memory_str = memory_str
        self.memory_num = memory_num
        self.memory_bool = []
        self.header_str = header_str
        self.header_num = header_num
        self.trace_str = [v for k, v in memory_str]
        self.trace_num = [v for k, v in memory_num]
        # For intermediate data frame
        self.rows = [("all_rows", rows)]

        self.cur_str = ""
        self.cur_strs = []
        self.cur_funcs = []

        self.must_have = must_have
        self.must_not_have = must_not_have

        self.row_counter = [1]

    def done(self):
        if self.memory_str_len == 0 and self.memory_num_len == 0 and \
                self.memory_bool_len == 0 and all([_ > 0 for _ in self.row_counter]):
            for funcs in self.must_have:
                if any([f in self.cur_funcs for f in funcs]):
                    continue
                else:
                    return False
            return True
        else:
            return False

    @property
    def tostring(self):
        print("memory_str:", self.memory_str)
        print("memory_num:", self.memory_num)
        print("header_str:", self.header_str)
        print("header_num:", self.header_num)
        print("trace:", self.cur_str)

    def concat(self, new_str, k):
        func = new_str.split('(')[0]
        self.cur_funcs.append(func)
        self.cur_strs.append(new_str)

    def exist(self, command):
        return command in self.cur_strs

    def clone(self, command, k):
        tmp = copy.deepcopy(self)
        tmp.concat(command, k)
        return tmp

    @property
    def memory_str_len(self):
        return len(self.memory_str)

    @property
    def memory_num_len(self):
        return len(self.memory_num)

    @property
    def tmp_memory_num_len(self):
        return len([_ for _ in self.memory_num if "tmp_" in _ and _ != "tmp_none"])
        # return len(self.memory_num)

    @property
    def tmp_memory_str_len(self):
        return len([_ for _ in self.memory_str if "tmp_" in _])

    @property
    def memory_bool_len(self):
        return len(self.memory_bool)

    @property
    def row_num(self):
        return len(self.rows) - 1

    @property
    def hash(self):
        return hash(frozenset(self.cur_strs))

    def append_result(self, command, r):
        self.cur_str = "{}={}".format(command, r)

    def append_bool(self, r):
        if self.cur_str != "":
            self.cur_str += ";{}".format(r)
        else:
            self.cur_str = "{}".format(r)

    def get_memory_str(self, i):
        return self.memory_str[i][1]

    def get_memory_num(self, i):
        return self.memory_num[i][1]

    def add_memory_num(self, header, val, command):
        if type(val) == type(1) or type(val) == type(1.2):
            val = val
        else:
            val = val.item()

        self.memory_num.append((header, val))
        self.trace_num.append(command)

    def add_memory_bool(self, header, val):
        if isinstance(val, bool):
            self.memory_bool.append((header, val))
        else:
            raise ValueError("type error: {}".format(type(val)))

    def add_memory_str(self, header, val, command):
        if isinstance(val, str):
            self.memory_str.append((header, val))
            self.trace_str.append(command)
        else:
            raise ValueError("type error: {}".format(type(val)))

    def add_header_str(self, header):
        self.header_str.append(header)

    def add_header_num(self, header):
        self.header_num.append(header)

    def add_rows(self, header, val):
        if isinstance(val, pandas.DataFrame):
            # for row_h, row in self.rows:
            #    if len(row) == len(val) and row.iloc[0][0] == val.iloc[0][0]:
            #        return
            if any([row_h == header for row_h, row in self.rows]):
                return
            self.rows.append((header, val.reset_index(drop=True)))
            self.row_counter.append(0)
        else:
            raise ValueError("type error")

    def inc_row_counter(self, i):
        self.row_counter[i] += 1

    def delete_memory_num(self, *args):
        new_mem_num = []
        new_trace_num = []
        for k in range(len(self.memory_num)):
            if k in args:
                continue
            else:
                new_mem_num.append(self.memory_num[k])
                new_trace_num.append(self.trace_num[k])

        self.memory_num = new_mem_num
        self.trace_num = new_trace_num

    def delete_memory_str(self, *args):
        new_mem_str = []
        new_trace_str = []
        for k in range(len(self.memory_str)):
            if k in args:
                continue
            else:
                new_mem_str.append(self.memory_str[k])
                new_trace_str.append(self.trace_str[k])

        self.memory_str = new_mem_str
        self.trace_str = new_trace_str

    def delete_memory_bool(self, *args):
        new_bool = []
        for k in range(len(self.memory_bool)):
            if k in args:
                continue
            else:
                new_bool.append(self.memory_bool[k])

        self.memory_bool= new_bool

    def check(self, *args):
        final = {}
        for arg in args:
            if arg == 'row':
                continue

            if arg == ['header_str', 'string']:
                if any([k is not None for k, v in self.memory_str]):
                    continue
                else:
                    return False

            if arg == ['header_num', 'number']:
                if any([k is not None for k, v in self.memory_num]):
                    continue
                else:
                    return False

            if arg == 'string':
                if len(self.memory_str) > 0:
                    continue
                else:
                    return False

            if arg == 'number':
                if len(self.memory_num) > 0:
                    continue
                else:
                    return False

            if arg == 'header_str':
                if len(self.header_str) > 0:
                    continue
                else:
                    return False

            if arg == 'header_num':
                if len(self.header_num) > 0:
                    continue
                else:
                    return False
        return True


def dynamic_programming(name, t, orig_sent, sent, tags, mem_str, mem_num, head_str, head_num, num=6, debug=False):
    must_have = []
    must_not_have = []
    for k, v in non_triggers.items():
        if isinstance(v[0], list):
            flags = []
            for v_sub in v:
                flag = False
                for trigger in v_sub:
                    if 'REG:' in trigger:
                        regex = trigger.replace('REG:', '')
                        if re.search(regex, sent):
                            flag = True
                            break
                    elif trigger in ['RBR', 'RBS', 'JJR', 'JJS']:
                        if trigger in tags:
                            flag = True
                            break
                    else:
                        if " " + trigger + " " in " " + sent + " ":
                            flag = True
                            break
                flags.append(flag)
            if not all(flags):
                must_not_have.append(k)
        else:
            flag = False
            for trigger in v:
                if 'REG:' in trigger:
                    regex = trigger.replace('REG:', '')
                    if re.search(regex, sent):
                        flag = True
                        break
                elif trigger in ['RBR', 'RBS', 'JJR', 'JJS']:
                    if trigger in tags:
                        flag = True
                        break
                else:
                    if " " + trigger + " " in " " + sent + " ":
                        flag = True
                        break
            if not flag:
                must_not_have.append(k)

    node = Node(memory_str=mem_str, memory_num=mem_num, rows=t,
                header_str=head_str, header_num=head_num, must_have=must_have, must_not_have=must_not_have)

    count_all = any([k == 'tmp_input' for k, v in mem_num])

    start_time = time.time()
    # The result storage
    finished = []
    hist = [[node]] + [[] for _ in range(num)]
    cache = {}

    def call(command, f, *args):
        if command not in cache:
            cache[command] = f(*args)
            return cache[command]
        else:
            return cache[command]

    start_time = time.time()
    for step in range(len(hist) - 1):
        # Iterate over father nodes
        saved_hash = []

        def conditional_add(tmp, path):
            success = False
            if tmp.hash not in saved_hash:
                path.append(tmp)
                saved_hash.append(tmp.hash)
                success = True
            return success

        for root in hist[step]:
            # Iterate over API
            for k, v in APIs.items():
                # propose candidates
                if k in root.must_not_have or not root.check(*v['argument']):
                    continue

                if v['output'] == 'row' and root.row_num >= 2:
                    continue
                if v['output'] == 'num' and root.tmp_memory_num_len >= 3:
                    continue
                if v['output'] == 'str' and root.tmp_memory_str_len >= 3:
                    continue
                if v['output'] == 'bool' and root.memory_bool_len >= 3:
                    continue
                if 'inc_' in k and 'inc' in root.cur_funcs:
                    continue
                """
                elif v['argument'] == ["header_num"]:
                    for l in range(len(root.header_num)):
                        command = v['tostr'](root.header_num[l])
                        if not root.exist(command):
                            tmp = root.clone(command)
                            returned = v['function'](root.header_num[l])
                            tmp.add_header_num(returned)
                            conditional_add(tmp, hist[i + 1])
                elif v['argument'] == ["header_str"]:
                    for l in range(len(root.header_str)):
                        command = v['tostr'](root.header_str[l])
                        if not root.exist(command):
                            tmp = root.clone(command)
                            returned = v['function'](root.header_str[l])
                            tmp.add_header_str(returned)
                            conditional_add(tmp, hist[i + 1])
                """
                # Incrementing/Decrementing/Whether is zero
                if v['argument'] == ["num"]:
                    for i, (h, va) in enumerate(root.memory_num):
                        if v['output'] == 'num':
                            if step == 0 and "tmp" in h:
                                command = v['tostr'](root.trace_num[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], va)
                                    tmp.add_memory_num(h, returned, returned)
                                    conditional_add(tmp, hist[step + 1])
                        elif v['output'] == 'bool':
                            if "tmp_" in h and "count" not in h:
                                command = v['tostr'](root.trace_num[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], va)
                                    tmp.delete_memory_num(i)
                                    if tmp.done():
                                        tmp.append_result(command, returned)
                                        finished.append((tmp, returned))
                                    else:
                                        tmp.add_memory_bool(command, returned)
                                        conditional_add(tmp, hist[step + 1])
                        elif v['output'] == 'none':
                            if step == 0 and "tmp" in h:
                                command = v['tostr'](root.trace_num[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_num(i)
                                    if tmp.done():
                                        continue
                                    else:
                                        conditional_add(tmp, hist[step + 1])
                        else:
                            raise ValueError("Returned Type Wrong")

                # Incrementing/Decrementing/Whether is none
                elif v['argument'] == ["str"]:
                    for i, (h, va) in enumerate(root.memory_str):
                        if v['output'] == 'str':
                            if step == 0:
                                if "tmp_" not in h:
                                    command = v['tostr'](root.trace_str[i])
                                    if not root.exist(command):
                                        tmp = root.clone(command, k)
                                        returned = call(command, v['function'], va)
                                        tmp.add_memory_str(h, returned, returned)
                                        conditional_add(tmp, hist[step + 1])
                        elif v['output'] == 'bool':
                            if k == "existing" and step == 0:
                                pass
                            elif k == "none" and "tmp_" in h:
                                pass
                            else:
                                continue
                            command = v['tostr'](root.trace_str[i])
                            if not root.exist(command):
                                tmp = root.clone(command, k)
                                returned = call(command, v['function'], va)
                                tmp.delete_memory_str(i)
                                if tmp.done():
                                    tmp.append_result(command, returned)
                                    finished.append((tmp, returned))
                                else:
                                    tmp.add_memory_bool(command, returned)
                                    conditional_add(tmp, hist[step + 1])
                        elif v['output'] == 'none':
                            if step == 0:
                                command = v['tostr'](root.trace_str[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_str(i)
                                    if tmp.done():
                                        continue
                                    else:
                                        conditional_add(tmp, hist[step + 1])
                        else:
                            raise ValueError("Returned Type Wrong")

                elif v['argument'] == ['strs']:
                    if step == 0:
                        if len(root.memory_str) > 1 and all(["tmp_" not in h for h, _ in root.memory_str]):
                            if len(set([h for h, _ in root.memory_str])) == 1:
                                command = v['tostr']([root.trace_str[i] for i in range(len(root.memory_str))])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_str(*range(len(root.memory_str)))
                                    returned = call(command, v['function'], [va for h, va in root.memory_str])
                                    if tmp.done():
                                        tmp.append_result(command, returned)
                                        finished.append((tmp, returned))
                                    else:
                                        tmp.add_memory_bool(command, returned)
                                        conditional_add(tmp, hist[step + 1])

                elif v['argument'] == ['row', 'header_str', 'str']:
                    for j, (row_h, row) in enumerate(root.rows):
                        for i, (h, va) in enumerate(root.memory_str):
                            if "tmp_" in h or len(row) == 1:
                                continue
                            for head in root.header_str:
                                if "; " + head + ";" in row_h:
                                    continue
                                command = v['tostr'](row_h, head, root.trace_str[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], row, head, va)
                                    if v['output'] == "bool":
                                        tmp.inc_row_counter(j)
                                        tmp.delete_memory_str(i)
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError("Returned Type Wrong")

                elif v['argument'] == ['row', 'header_num', 'num']:
                    for j, (row_h, row) in enumerate(root.rows):
                        for i, (h, va) in enumerate(root.memory_num):
                            if "tmp_" in h or len(row) == 1:
                                continue
                            for head in root.header_num:
                                if "; " + head + ";" in row_h:
                                    continue
                                command = v['tostr'](row_h, head, root.trace_num[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], row, head, va)
                                    if v['output'] == "bool":
                                        tmp.inc_row_counter(j)
                                        tmp.delete_memory_num(i)
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError("Returned Type Wrong")

                elif v['argument'] == ['bool', 'bool']:
                    if root.memory_bool_len < 2:
                        continue
                    else:
                        for l in range(0, root.memory_bool_len - 1):
                            for m in range(l + 1, root.memory_bool_len):
                                command = v['tostr'](root.memory_bool[l][0], root.memory_bool[m][0])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], root.memory_bool[l]
                                                    [1], root.memory_bool[m][1])
                                    if v['output'] == "bool":
                                        tmp.delete_memory_bool(l, m)
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError("Returned Type Wrong")

                elif v['argument'] == ['row']:
                    for j, (row_h, row) in enumerate(root.rows):
                        if k == "count":
                            if row_h.startswith('filter'):
                                pass
                            elif row_h == "all_rows":
                                if count_all:
                                    pass
                                else:
                                    continue
                        elif k == "only":
                            if not row_h.startswith('filter'):
                                continue
                        else:
                            if not row_h == "all_rows":
                                continue
                        command = v['tostr'](row_h)
                        if not root.exist(command):
                            tmp = root.clone(command, k)
                            tmp.inc_row_counter(j)
                            returned = call(command, v['function'], row)
                            if v['output'] == 'num':
                                tmp.add_memory_num("tmp_count", returned, command)
                            elif v['output'] == 'row':
                                tmp.add_rows(command, returned)
                                conditional_add(tmp, hist[step + 1])
                            elif v['output'] == 'bool':
                                if tmp.done():
                                    tmp.append_result(command, returned)
                                    finished.append((tmp, returned))
                                elif tmp.memory_bool_len < 2:
                                    tmp.add_memory_bool(command, returned)
                                    conditional_add(tmp, hist[step + 1])
                            else:
                                raise ValueError("error, out of scope")
                            conditional_add(tmp, hist[step + 1])

                elif v['argument'] == ['row', 'row', 'row']:
                    if len(root.rows) < 3:
                        continue
                    _, all_rows = root.rows[0]
                    for i in range(1, len(root.rows) - 1):
                        for j in range(i + 1, len(root.rows)):
                            if v['output'] == 'bool':
                                if len(root.rows[i][1]) != 1 or len(root.rows[j][1]) != 1:
                                    continue
                                command = v['tostr'](root.rows[i][0], root.rows[j][0])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(i)
                                    tmp.inc_row_counter(j)
                                    returned = call(command, v['function'], all_rows, root.rows[i][1], root.rows[j][1])
                                    if returned is not None:
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                            else:
                                raise ValueError("error, out of scope")

                elif v['argument'] == ['row', 'row']:
                    if len(root.rows) < 2:
                        continue
                    for i in range(len(root.rows) - 1):
                        for j in range(i + 1, len(root.rows)):
                            if v['output'] == 'bool':
                                if len(root.rows[i][1]) != 1 and len(root.rows[j][1]) != 1:
                                    continue
                                command = v['tostr'](root.rows[i][0], root.rows[j][0])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(i)
                                    tmp.inc_row_counter(j)
                                    returned = call(command, v['function'], root.rows[i][1], root.rows[j][1])
                                    if returned is not None:
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                            else:
                                raise ValueError("error, out of scope")

                elif v['argument'] == ['row', 'header_num']:
                    if "hop" in k:
                        for j, (row_h, row) in enumerate(root.rows):
                            if len(row) != 1:
                                continue
                            for l in range(len(root.header_num)):
                                command = v['tostr'](row_h, root.header_num[l])
                                if "; " + root.header_num[l] + ";" in row_h:
                                    continue
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    returned = call(command, v['function'], row, root.header_num[l])
                                    if v['output'] == 'num':
                                        tmp.add_memory_num("tmp_" + root.header_num[l], returned, command)
                                        conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError("error, output of scope")
                    else:
                        for j, (row_h, row) in enumerate(root.rows):
                            if len(row) == 1:
                                continue
                            for l in range(len(root.header_num)):
                                command = v['tostr'](row_h, root.header_num[l])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    # It does not make sense to do min/max over one line
                                    if any([_ in k for _ in ['max', 'min']]) and len(row) == 1:
                                        continue

                                    returned = call(command, v['function'], row, root.header_num[l])
                                    if v['output'] == 'num':
                                        tmp.add_memory_num("tmp_" + root.header_num[l], returned, command)
                                        conditional_add(tmp, hist[step + 1])
                                    elif v['output'] == 'row':
                                        if len(returned) > 0:
                                            tmp.add_rows(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                        else:
                                            continue
                                    else:
                                        raise ValueError("error, output of scope")

                elif v['argument'] == ['row', 'header_str']:
                    if "most_freq" in k:
                        row_h, row = root.rows[0]
                        for l in range(len(root.header_str)):
                            command = v['tostr'](row_h, root.header_str[l])
                            if not root.exist(command):
                                tmp = root.clone(command, k)
                                returned = call(command, v['function'], row, root.header_str[l])
                                if v['output'] == 'str':
                                    if returned is not None:
                                        tmp.add_memory_str("tmp_" + root.header_str[l], returned, command)
                                        conditional_add(tmp, hist[step + 1])
                                else:
                                    raise ValueError("error, output of scope")
                    elif "hop" in k:
                        for j, (row_h, row) in enumerate(root.rows):
                            if len(row) != 1:
                                continue
                            for l in range(len(root.header_str)):
                                if "; " + root.header_str[l] + ";" in row_h:
                                    continue
                                command = v['tostr'](row_h, root.header_str[l])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    returned = call(command, v['function'], row, root.header_str[l])
                                    if v['output'] == 'str':
                                        if isinstance(returned, str):
                                            if is_number(returned):
                                                returned = float(returned)
                                                tmp.add_memory_num("tmp_" + root.header_str[l], returned, command)
                                            else:
                                                tmp.add_memory_str("tmp_" + root.header_str[l], returned, command)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError("error, output of scope")
                    else:
                        for j, (row_h, row) in enumerate(root.rows):
                            if len(row) == 1:
                                continue
                            for l in range(len(root.header_str)):
                                command = v['tostr'](row_h, root.header_str[l])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    returned = call(command, v['function'], row, root.header_str[l])
                                    if v['output'] == 'str':
                                        if isinstance(returned, str):
                                            tmp.add_memory_str("tmp_" + root.header_str[l], returned, command)
                                            conditional_add(tmp, hist[step + 1])
                                    elif v['output'] == 'row':
                                        if len(returned) > 0:
                                            tmp.add_rows(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                        else:
                                            continue
                                    elif v['output'] == 'num':
                                        tmp.add_memory_num("tmp_count", returned, command)
                                        conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError("error, output of scope")

                elif v['argument'] == ['num', 'num']:
                    if root.memory_num_len < 2:
                        continue
                    for l in range(0, root.memory_num_len - 1):
                        for m in range(l + 1, root.memory_num_len):
                            if 'tmp_' in root.memory_num[l][0] or 'tmp_' in root.memory_num[m][0]:
                                if ("tmp_input" == root.memory_num[l][0] and "tmp_" not in root.memory_num[m][0]) or \
                                        ("tmp_input" == root.memory_num[m][0] and "tmp_" not in root.memory_num[l][0]):
                                    continue
                                elif root.memory_num[l][0] == root.memory_num[m][0] == "tmp_input":
                                    continue
                            else:
                                continue

                            type_l = root.memory_num[l][0].replace('tmp_', '')
                            type_m = root.memory_num[m][0].replace('tmp_', '')
                            if v['output'] == 'num':
                                if type_l == type_m:
                                    # Two direction:
                                    for dir1, dir2 in zip([l, m], [m, l]):
                                        command = v['tostr'](root.trace_num[dir1], root.trace_num[dir2])               
                                        tmp = root.clone(command, k)
                                        tmp.delete_memory_num(dir1, dir2)
                                        returned = call(command, v['function'],
                                                        root.get_memory_num(dir1), root.get_memory_num(dir2))
                                        tmp.add_memory_num("tmp_" + root.memory_num[dir1][0], returned, command)
                                        r = conditional_add(tmp, hist[step + 1])

                            elif v['output'] == 'bool':
                                if type_l == type_m or (type_l == "input" or type_m == "input"):
                                    pass
                                else:
                                    continue

                                if type_l == "count" and type_m == "input" or type_m == "count" and type_l == "input":
                                    if max(root.get_memory_num(l), root.get_memory_num(m)) > len(root.rows[0][1]):
                                        continue

                                # Two direction:
                                for dir1, dir2 in zip([l, m], [m, l]):
                                    command = v['tostr'](root.trace_num[dir1], root.trace_num[dir2])
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_num(dir1, dir2)
                                    returned = call(command, v['function'], root.get_memory_num(dir1), root.get_memory_num(dir2))
                                    if tmp.done():
                                        tmp.append_result(command, returned)
                                        finished.append((tmp, returned))
                                    elif tmp.memory_bool_len < 2:
                                        tmp.add_memory_bool(command, returned)
                                        conditional_add(tmp, hist[step + 1])
                            else:
                                raise ValueError("error, output of scope")

                elif v['argument'] == ['str', 'str']:
                    if root.memory_str_len < 2:
                        continue
                    for l in range(0, root.memory_str_len - 1):
                        for m in range(l + 1, root.memory_str_len):
                            if 'tmp_' not in root.memory_str[l][0] and 'tmp_' not in root.memory_str[m][0]:
                                continue
                            type_l = root.memory_str[l][0].replace('tmp_', '')
                            type_m = root.memory_str[m][0].replace('tmp_', '')
                            if type_l == type_m:
                                command = v['tostr'](root.trace_str[l], root.trace_str[m])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_str(l, m)
                                    if v['output'] == 'bool':
                                        returned = call(command, v['function'],
                                                        root.get_memory_str(m), root.get_memory_str(l))
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError("error, output of scope")

                elif v['argument'] == ['row', ['header_str', 'str']]:
                    for j, (row_h, row) in enumerate(root.rows):
                        for i, (h, va) in enumerate(root.memory_str):
                            if "tmp_" not in h:
                                command = v['tostr'](row_h, h, root.trace_str[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    tmp.delete_memory_str(tmp.memory_str.index((h, va)))
                                    returned = call(command, v['function'], row, h, va)
                                    if v['output'] == 'row':
                                        if len(returned) > 0:
                                            tmp.add_rows(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                        else:
                                            continue
                                    elif v['output'] == 'bool':
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError('error, output of scope')

                elif v['argument'] == ['row', ['header_num', 'num']]:
                    for j, (row_h, row) in enumerate(root.rows):
                        # It does not make sense to do filter/all operation on one row
                        if len(row) == 1:
                            continue
                        for i, (h, va) in enumerate(root.memory_num):
                            if "tmp_" not in h:
                                command = v['tostr'](row_h, h, root.trace_num[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    tmp.delete_memory_num(tmp.memory_num.index((h, va)))
                                    returned = call(command, v['function'], row, h, va)
                                    if v['output'] == 'row':
                                        if len(returned) > 0:
                                            tmp.add_rows(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                        else:
                                            continue
                                    elif v['output'] == 'bool':
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError('error, output of scope')

                elif v['argument'] == [['header_str', 'str'], ['header_num', 'num']]:
                    if not root.memory_str_len or not root.memory_num_len:
                        continue
                    row_h, row = root.rows[0]
                    for i, (h1, va1) in enumerate(root.memory_str):
                        for j, (h2, va2) in enumerate(root.memory_num):
                            if "tmp_" not in h1 and "tmp_" not in h2:
                                command = v['tostr'](h1, root.trace_str[i], h2, root.trace_num[j])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_str(i)
                                    tmp.delete_memory_num(j)
                                    returned = call(command, v['function'], row, h1, va1, h2, va2)
                                    if v['output'] == 'bool':
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError('error, output of scope')

                elif v['argument'] == [['header_str', 'str'], ['header_str', 'str']]:
                    if root.memory_str_len < 2:
                        continue
                    row_h, row = root.rows[0]
                    for l in range(len(root.memory_str) - 1):
                        for m in range(l + 1, len(root.memory_str)):
                            h1, va1 = root.memory_str[l]
                            h2, va2 = root.memory_str[m]
                            if "tmp_" not in h1 and "tmp_" not in h2 and (h1 != h2):
                                command = v['tostr'](h1, root.trace_str[l], h2, root.trace_str[m])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_str(l, m)
                                    returned = call(command, v['function'], row, h1, va1, h2, va2)
                                    if v['output'] == 'bool':
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError('error, output of scope')

                elif v['argument'] == [['header_num', 'num'], ['header_num', 'num']]:
                    if root.memory_num_len < 2:
                        continue
                    row_h, row = root.rows[0]
                    for l in range(len(root.memory_num) - 1):
                        for m in range(l + 1, len(root.memory_num)):
                            h1, va1 = root.memory_num[l]
                            h2, va2 = root.memory_num[m]
                            if ("tmp_" not in h1 and "tmp_" not in h2) and (h1 != h2):
                                command = v['tostr'](h1, root.trace_num[l], h2, root.trace_num[m])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_num(l, m)
                                    returned = call(command, v['function'], row, h1, va1, h2, va2)
                                    if v['output'] == 'bool':
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                else:
                    raise ValueError(k + ": error")

        if len(finished) > 100 or time.time() - start_time > 30:
            break

    return (name, orig_sent, sent, [_[0].cur_str for _ in finished])


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def augment(s):
    recover_dict = {}
    if 'first' in s:
        s.append("1st")
        recover_dict[s[-1]] = 'first'
    elif 'second' in s:
        s.append("2nd")
        recover_dict[s[-1]] = 'second'
    elif 'third' in s:
        s.append("3rd")
        recover_dict[s[-1]] = 'third'
    elif 'fourth' in s:
        s.append("4th")
        recover_dict[s[-1]] = 'fourth'
    elif 'fifth' in s:
        s.append("5th")
        recover_dict[s[-1]] = 'fifth'
    elif 'sixth' in s:
        s.append("6th")
        recover_dict[s[-1]] = 'sixth'

    for i in range(1, 10):
        if "0" + str(i) in s:
            s.append(str(i))
            recover_dict[s[-1]] = "0" + str(i)

    if 'crowd' in s or 'attendance' in s:
        s.append("people")
        recover_dict[s[-1]] = 'crowd'
        s.append("audience")
        recover_dict[s[-1]] = 'crowd'

    if any([_ in months_a + months_b for _ in s]):
        for i in range(1, 32):
            if str(i) in s:
                if i % 10 == 1:
                    s.append(str(i) + "st")
                elif i % 10 == 2:
                    s.append(str(i) + "nd")
                elif i % 10 == 3:
                    s.append(str(i) + "rd")
                else:
                    s.append(str(i) + "th")
                recover_dict[s[-1]] = str(i)

        for k in a2b:
            if k in s:
                s.append(a2b[k])
                recover_dict[s[-1]] = k

        for k in b2a:
            if k in s:
                s.append(b2a[k])
                recover_dict[s[-1]] = k

    return s, recover_dict


def replace_useless(s):
    s = s.replace(',', '')
    s = s.replace('.', '')
    s = s.replace('/', '')
    s = s.replace('  ', '')
    return s


def get_closest(inp, string, indexes, tabs, threshold):
    if string in stop_words:
        return None

    dist = 10000
    rep_string = replace_useless(string)
    len_string = len(rep_string.split())

    minimum = []
    for index in indexes:
        entity = replace_useless(tabs[index[0]][index[1]])
        len_tab = len(entity.split())
        if abs(len_tab - len_string) < dist:
            minimum = [index]
            dist = abs(len_tab - len_string)
        elif abs(len_tab - len_string) == dist:
            minimum.append(index)

    vocabs = []
    for s in rep_string.split(' '):
        vocabs.append(vocab.get(s, 10000))

    # Whether contain rare words
    if dist == 0:
        return minimum[0]

    # String Length
    feature = [len_string]
    # Proportion
    feature.append(-dist / (len_string + dist + 0.) * 4)
    if any([(s.isdigit() and int(s) < 100) for s in rep_string.split()]):
        feature.extend([0, 0])
    else:
        # Quite rare words
        if max(vocabs) > 800:
            feature.append(1)
        else:
            feature.append(-1)
        # Whether contain super rare words
        if max(vocabs) > 2000:
            feature.append(3)
        else:
            feature.append(0)
    # Whether it is only a word
    if len_string > 1:
        feature.append(1)
    else:
        feature.append(0)
    # Whether candidate has only one
    if len(indexes) == 1:
        feature.append(1)
    else:
        feature.append(0)
    # Whether cover over half of it
    if len_string > dist:
        feature.append(1)
    else:
        feature.append(0)

    # Whether contains alternative
    cand = replace_useless(tabs[minimum[0][0]][minimum[0][1]])
    if '(' in cand and ')' in cand:
        feature.append(2)
    else:
        feature.append(0)
    # Match more with the header
    if minimum[0][0] == 0:
        feature.append(2)
    else:
        feature.append(0)
    # Whether it is a month
    if any([" " + _ + " " in " " + rep_string + " " for _ in months_a + months_b]):
        feature.append(5)
    else:
        feature.append(0)

    # Whether it matches against the candidate
    if rep_string in cand:
        feature.append(0)
    else:
        feature.append(-5)

    if sum(feature) > threshold:
        if len(minimum) > 1:
            if minimum[0][0] > 0:
                return [-2, minimum[0][1]]
            else:
                return minimum[0]
        else:
            return minimum[0]
    else:
        return None

def split_prog(string, merge_words=False):
    array = []
    tmp = ''
    if '=True' in string:
        result = True
        string.rstrip('=True')
    elif '=False' in string:
        result = False
        string.rstrip('=False')
    else:
        result = None

    for s in string:
        if s == '{' or s == ';' or s == '}':
            if tmp:
                array.append(tmp)
                tmp = ''
            array.append(s)
        elif s == ' ':
            if tmp:
                if merge_words:
                    tmp += s
                    continue
                else:
                    array.append(tmp)
                    tmp = ''
        else:
            tmp += s

    if result is not None:
        return array, result
    else:
        return array

def replace_number(string):
    string = re.sub(r'(\b)one(\b)', r'\g<1>1\g<2>', string)
    string = re.sub(r'(\b)two(\b)', '\g<1>2\g<2>', string)
    string = re.sub(r'(\b)three(\b)', '\g<1>3\g<2>', string)
    string = re.sub(r'(\b)four(\b)', '\g<1>4\g<2>', string)
    string = re.sub(r'(\b)five(\b)', '\g<1>5\g<2>', string)
    string = re.sub(r'(\b)six(\b)', '\g<1>6\g<2>', string)
    string = re.sub(r'(\b)seven(\b)', '\g<1>7\g<2>', string)
    string = re.sub(r'(\b)eight(\b)', '\g<1>8\g<2>', string)
    string = re.sub(r'(\b)nine(\b)', '\g<1>9\g<2>', string)
    string = re.sub(r'(\b)ten(\b)', '\g<1>10\g<2>', string)
    string = re.sub(r'(\b)eleven(\b)', '\g<1>11\g<2>', string)
    string = re.sub(r'(\b)twelve(\b)', '\g<1>12\g<2>', string)
    string = re.sub(r'(\b)thirteen(\b)', '\g<1>13\g<2>', string)
    string = re.sub(r'(\b)fourteen(\b)', '\g<1>14\g<2>', string)
    string = re.sub(r'(\b)fifteen(\b)', '\g<1>15\g<2>', string)
    string = re.sub(r'(\b)sixteen(\b)', '\g<1>16\g<2>', string)
    string = re.sub(r'(\b)seventeen(\b)', '\g<1>17\g<2>', string)
    string = re.sub(r'(\b)eighteen(\b)', '\g<1>18\g<2>', string)
    string = re.sub(r'(\b)nineteen(\b)', '\g<1>19\g<2>', string)
    string = re.sub(r'(\b)twenty(\b)', '\g<1>20\g<2>', string)
    
    string = string.replace('a hundred', '100')
    string = string.replace('a thousand', '1000')

    string = re.sub(r'\b([0-9]+) hundred', r'\g<1>00', string)
    string = re.sub(r'\b([0-9]+) thousand', r'\g<1>000', string)

    return string


def replace(w, transliterate):
    if w in transliterate:
        return transliterate[w]
    else:
        return w


def intersect(w_new, w_old):
    new_set = []
    for w_1 in w_new:
        for w_2 in w_old:
            if w_1[:2] == w_2[:2] and w_1[2] > w_2[2]:
                new_set.append(w_2)
    return new_set


def recover(buf, recover_dict, content):
    if len(recover_dict) == 0:
        return buf
    else:
        new_buf = []
        for w in buf.split(' '):
            if w not in content:
                new_buf.append(recover_dict.get(w, w))
            else:
                new_buf.append(w)
        return ' '.join(new_buf)


def postprocess(inp, pos_tags, backbone, trans_backbone, transliterate, tabs, recover_dicts, repeat, threshold=1.0):
    new_str = []
    new_tags = []
    buf = ""
    pos_buf = []
    last = set()
    prev_closest = []

    for w, p in zip(inp, pos_tags):
        if (w in backbone) and ((" " + w + " " in " " + buf + " " and w in repeat) or (" " + w + " " not in " " + buf + " ")):
            if buf == "":
                last = set(backbone[w])
                buf = w
                pos_buf.append(p)
            else:
                proposed = set(backbone[w]) & last
                if not proposed:
                    closest = get_closest(inp, buf, last, tabs, threshold)
                    if closest:
                        buf = '#{};{},{}#'.format(recover(buf, recover_dicts[closest[0]][closest[1]],
                                                          tabs[closest[0]][closest[1]]), closest[0], closest[1])

                    new_str.append(buf)
                    if buf.startswith("#"):
                        new_tags.append('ENT')
                    else:
                        new_tags.extend(pos_buf)
                    pos_buf = []
                    buf = w
                    last = set(backbone[w])
                    pos_buf.append(p)
                else:
                    last = proposed
                    buf += " " + w
                    pos_buf.append(p)

        elif w in trans_backbone and ((" " + w + " " in " " + buf + " " and w in repeat) or (" " + w + " " not in " " + buf + " ")):
            if buf == "":
                last = set(trans_backbone[w])
                buf = transliterate[w]
                pos_buf.append(p)
            else:
                proposed = set(trans_backbone[w]) & last
                if not proposed:
                    closest = get_closest(inp, buf, last, tabs, threshold)
                    if closest:
                        buf = '#{};{},{}#'.format(recover(buf, recover_dicts[closest[0]][closest[1]],
                                                          tabs[closest[0]][closest[1]]), closest[0], closest[1])
                    new_str.append(buf)
                    if buf.startswith("#"):
                        new_tags.append('ENT')
                    else:
                        new_tags.extend(pos_buf)
                    pos_buf = []
                    buf = transliterate[w]
                    last = set(trans_backbone[w])
                    pos_buf.append(p)
                else:
                    buf += " " + transliterate[w]
                    last = proposed
                    pos_buf.append(p)

        else:
            if buf != "":
                closest = get_closest(inp, buf, last, tabs, threshold)
                if closest:
                    buf = '#{};{},{}#'.format(recover(buf, recover_dicts[closest[0]][closest[1]],
                                                      tabs[closest[0]][closest[1]]), closest[0], closest[1])
                new_str.append(buf)
                if buf.startswith("#"):
                    new_tags.append('ENT')
                else:
                    new_tags.extend(pos_buf)
                pos_buf = []

            buf = ""
            last = set()
            new_str.append(w)
            new_tags.append(p)

    if buf != "":
        closest = get_closest(inp, buf, last, tabs, threshold)
        if closest:
            buf = '#{};{},{}#'.format(recover(buf, recover_dicts[closest[0]][closest[1]],
                                              tabs[closest[0]][closest[1]]), closest[0], closest[1])
        new_str.append(buf)
        if buf.startswith("#"):
            new_tags.append('ENT')
        else:
            new_tags.extend(pos_buf)
        pos_buf = []

    return " ".join(new_str), " ".join(new_tags)


def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def isnumber(string):
    return string in [numpy.dtype('int64'), numpy.dtype('int32'), numpy.dtype('float32'), numpy.dtype('float64')]

def list2tuple(inputs):
    mem = []
    for s in inputs:
        mem.append(tuple(s))
    return mem

def split(string, option):
    if option == "row":
        return string.split(',')[0]
    else:
        return string.split(',')[1]


def merge_strings(string, tags=None):
    buff = ""
    inside = False
    words = []

    for c in string:
        if c == "#" and not inside:
            inside = True
            buff += c
        elif c == "#" and inside:
            inside = False
            buff += c
            words.append(buff)
            buff = ""
        elif c == " " and not inside:
            if buff:
                words.append(buff)
            buff = ""
        elif c == " " and inside:
            buff += c
        else:
            buff += c

    if buff:
        words.append(buff)

    tags = tags.split(' ')
    assert len(words) == len(tags), "{} and {}".format(words, tags)

    i = 0
    while i < len(words):
        if i < 2:
            i += 1
        elif words[i].startswith('#') and (not words[i - 1].startswith('#')) and words[i - 2].startswith('#'):
            if is_number(words[i].split(';')[0][1:]) and is_number(words[i - 2].split(';')[0][1:]):
                i += 1
            else:
                prev_idx = words[i - 2].split(';')[1][:-1].split(',')
                cur_idx = words[i].split(';')[1][:-1].split(',')
                if cur_idx == prev_idx or (prev_idx[0] == '-2' and prev_idx[1] == cur_idx[1]):
                    position = "{},{}".format(cur_idx[0], cur_idx[1])
                    candidate = words[i - 2].split(';')[0] + " " + words[i].split(';')[0][1:] + ";" + position + "#"
                    words[i] = candidate
                    del words[i - 1]
                    del tags[i - 1]
                    i -= 1
                    del words[i - 1]
                    del tags[i - 1]
                    i -= 1
                else:
                    i += 1
        else:
            i += 1

    return " ".join(words), " ".join(tags)


class Parser(object):
    def __init__(self, folder, lemmatize_verbs=True):
        self.folder = folder

        with open('data/table_to_page.json') as f:
            files = json.load(f)
            self.title_mapping = {k:v[0] for k, v in files.items()}

        if lemmatize_verbs:
            self.tag_dict = {"JJ": wordnet.ADJ,
                        "NN": wordnet.NOUN,
                        "NNS": wordnet.NOUN,
                        "NNP": wordnet.NOUN,
                        "NNPS": wordnet.NOUN,
                        "VB": wordnet.VERB,
                        "VBD": wordnet.VERB,
                        "VBG": wordnet.VERB,
                        "VBN": wordnet.VERB,
                        "VBP": wordnet.VERB,
                        "VBZ": wordnet.VERB,
                        "RB": wordnet.ADV,
                        "RP": wordnet.ADV}
        else:
            self.tag_dict = {"JJ": wordnet.ADJ,
                        "NN": wordnet.NOUN,
                        "NNS": wordnet.NOUN,
                        "NNP": wordnet.NOUN,
                        "NNPS": wordnet.NOUN,
                        "RB": wordnet.ADV,
                        "RP": wordnet.ADV}

        self.lemmatizer = WordNetLemmatizer()

    def get_lemmatize(self, words, return_pos):
        recover_dict = {}
        words = nltk.word_tokenize(words)
        words = [word for word in words if word]
        pos_tags = [_[1] for _ in nltk.pos_tag(words)]
        word_roots = []
        for w, p in zip(words, pos_tags):
            if is_ascii(w) and p in self.tag_dict:
                lemm = self.lemmatizer.lemmatize(w, self.tag_dict[p])
                if lemm != w:
                    recover_dict[lemm] = w
                word_roots.append(lemm)
            else:
                word_roots.append(w)
        if return_pos:
            return word_roots, recover_dict, pos_tags
        else:
            return word_roots, recover_dict

    def get_table(self, table_name):
        file = os.path.join(self.folder, table_name)
        table = pandas.read_csv(file, delimiter="#")
        return table

    def entity_link(self, table_name, lemmatized_sent, pos_tags):
        backbone = {}
        trans_backbone = {}
        transliterate = {}
        tabs = []
        recover_dicts = []
        repeat = set()
        table_title = self.title_mapping[table_name]

        with open(os.path.join(self.folder, table_name), 'r') as f:
            for k, _ in enumerate(f.readlines()):
                tabs.append([])
                recover_dicts.append([])
                for l, w in enumerate(_.strip().split('#')):
                    tabs[-1].append(w)
                    if len(w) > 0:
                        lemmatized_w, recover_dict = self.get_lemmatize(w, False)
                        lemmatized_w, new_dict = augment(lemmatized_w)
                        recover_dict.update(new_dict)
                        recover_dicts[-1].append(recover_dict)
                        for i, sub in enumerate(lemmatized_w):
                            if sub not in backbone:
                                backbone[sub] = [(k, l)]
                                if not is_ascii(sub):
                                    trans_backbone[unidecode(sub)] = [(k, l)]
                                    transliterate[unidecode(sub)] = sub
                            else:
                                if (k, l) not in backbone[sub]:
                                    backbone[sub].append((k, l))
                                else:
                                    if sub not in months_a + months_b:
                                        repeat.add(sub)
                                if not is_ascii(sub):
                                    trans_backbone[unidecode(sub)].append((k, l))
                                    transliterate[unidecode(sub)] = sub

                        for i, sub in enumerate(w.split(' ')):
                            if sub not in backbone:
                                backbone[sub] = [(k, l)]
                                if not is_ascii(sub):
                                    trans_backbone[unidecode(sub)] = [(k, l)]
                                    transliterate[unidecode(sub)] = sub
                            else:
                                if (k, l) not in backbone[sub]:
                                    backbone[sub].append((k, l))
                                if not is_ascii(sub):
                                    trans_backbone[unidecode(sub)].append((k, l))
                                    transliterate[unidecode(sub)] = sub
                    else:
                        recover_dicts[-1].append({})

        # Masking the caption
        captions, _ = self.get_lemmatize(table_title.strip(), False)
        for i, w in enumerate(captions):
            if w not in backbone:
                backbone[w] = [(-1, -1)]
            else:
                backbone[w].append((-1, -1))

        tabs.append([" ".join(captions)])

        results = []

        sent, tags = postprocess(lemmatized_sent, pos_tags, backbone, trans_backbone,
                                 transliterate, tabs, recover_dicts, repeat, threshold=1.0)
        
        if re.search(r',[0-9]+#', sent):
            sent, tags = postprocess(lemmatized_sent, pos_tags, backbone, trans_backbone,
                                     transliterate, tabs, recover_dicts, repeat, threshold=0.0)

        sent, tags = merge_strings(sent, tags)

        return sent, tags

    def initialize_buffer(self, table_name, sent, pos_tag, raw_sent):
        count = 0
        preprocessed = []

        t = self.get_table(table_name)
        cols = t.columns
        mapping = {i: "num" if isnumber(t) else "str" for i, t in enumerate(t.dtypes)}

        count += 1
        inside = False
        position = False
        masked_sent = ''
        position_buf, mention_buf = '', ''
        mem_num, head_num, mem_str, head_str = [], [], [], []
        ent_index = 0
        ent2content = {}
        for n in range(len(sent)):
            if sent[n] == '#':
                if position:
                    if position_buf.startswith('0'):
                        idx = int(split(position_buf, "col"))
                        if mapping[idx] == 'num':
                            if cols[idx] not in head_num:
                                head_num.append(cols[idx])
                        else:
                            if cols[idx] not in head_str:
                                head_str.append(cols[idx])
                    else:
                        row = int(split(position_buf, "row"))
                        idx = int(split(position_buf, "col"))
                        if idx == -1:
                            pass
                        else:
                            if mapping[idx] == 'num':
                                if mention_buf.isdigit():
                                    mention_buf = int(mention_buf)
                                else:
                                    try:
                                        mention_buf = float(mention_buf)
                                    except Exception:
                                        pass
                                val = (cols[idx], mention_buf)
                                if val not in mem_num:
                                    mem_num.append(val)
                            else:
                                if len(fuzzy_match(t, cols[idx], mention_buf)) == 0:
                                    val = (cols[idx], mention_buf)
                                else:
                                    val = (cols[idx], mention_buf)
                                if val not in mem_str:
                                    mem_str.append(val)
                    # Direct matching
                    masked_sent += "<ENTITY{}>".format(ent_index)
                    ent2content["<ENTITY{}>".format(ent_index)] = str(mention_buf)
                    ent_index += 1
                    # Reset the buffer
                    position_buf = ""
                    mention_buf = ""
                    inside = False
                    position = False
                else:
                    inside = True
            elif sent[n] == ';':
                position = True
            else:
                if position:
                    position_buf += sent[n]
                elif inside:
                    mention_buf += sent[n]
                else:
                    masked_sent += sent[n]

        tokens = masked_sent.split()
        new_tokens = []
        for i in range(len(tokens)):
            _ = tokens[i]
            if _.isdigit():
                num = int(_)
            elif '.' in tokens[i]:
                try:
                    num = float(_)
                except Exception:
                    new_tokens.append(_)
                    continue
            else:
                new_tokens.append(_)
                continue

            # Reason whether the number is a comparison or a count
            compare_features = []
            count_features = []

            if tokens[i - 1] in months_b + months_a:
                compare_features.append(-6)
                count_features.append(-6)
            else:
                compare_features.append(0)
                count_features.append(0)

            if any([_ in tokens for _ in ["than", "over", "more", "less", "below", "above", 'least']]):
                compare_features.append(2)
            else:
                compare_features.append(-2)

            if any([_ in pos_tag for _ in ["RBR", "JJR"]]):
                compare_features.append(2)
            else:
                compare_features.append(-2)

            if isinstance(num, int):
                count_features.append(0)
                compare_features.append(0)
            else:
                count_features.append(-6)
                compare_features.append(4)

            if num > 50:
                count_features.append(-4)
                if num > 1900 and num < 2020:
                    compare_features.append(-2)
                else:
                    compare_features.append(2)
            else:
                if num > len(t):
                    compare_features.append(2)
                    count_features.append(-4)
                else:
                    compare_features.append(-1)
                    count_features.append(2)

            if i > 1 and tokens[i - 1] in ['be', 'are', 'is'] and tokens[i - 2] == 'there':
                count_features.append(4)
                compare_features.append(-4)
            elif i + 1 < len(tokens) and tokens[i + 1] == 'of':
                count_features.append(4)
                compare_features.append(-4)
            else:
                count_features.append(-1)
                compare_features.append(1)

            #if len(head_num) > 0:
            #    features.append(1)
            #else:
            #    features.append(0)
            if any([_ not in map(lambda x: x[0], mem_num) for _ in head_num]):
                compare_features.append(2)
                count_features.append(0)

            # Finally reaching a conclusion
            compare_score = sum(compare_features)
            count_score = sum(count_features)

            if compare_score >= count_score:
                if compare_score > 0:
                    new_tokens.append("<COMPUTE{}>".format(ent_index))
                    ent2content["<COMPUTE{}>".format(ent_index)] = _
                    ent_index += 1
                    if head_num:
                        for h in head_num:
                            if any([_[0] == h for _ in mem_num]):
                                continue
                            else:
                                mem_num.append((h, num))
                    else:
                        for col_idx, k in zip(range(len(cols) - 1, -1, -1), cols[::-1]):
                            if mapping[col_idx] == 'num':
                                mem_num.append((k, num))                                
                                head_num.append(k)
                                break                  
                else:
                    new_tokens.append(_)
                    continue
            else:
                if count_score > 0:
                    new_tokens.append("<COUNT{}>".format(ent_index))
                    ent2content["<COUNT{}>".format(ent_index)] = _
                    ent_index += 1
                    mem_num.append(("tmp_input", num))
                else:
                    new_tokens.append(_)
                    continue                    
        
        # Correct some wrongly linked count
        to_delete = []
        for index, (k, v) in enumerate(mem_num):
            if k != 'tmp_input' and isinstance(v, int) and v < 10:
                if 'there be {} '.format(v) in raw_sent or 'there are {} '.format(v) in raw_sent or\
                            'there is {} '.format(v) in raw_sent or ' {} of '.format(v) in raw_sent or raw_sent.startswith(str(v) + " "):
                    mem_num[index] = ('tmp_input', v)
                    for k, content in ent2content.items():
                        if content == str(v):
                            new_k = re.sub(r'<[^0-9]+([0-9]+)>', r'<COUNT\1>', k)
                            ent2content[new_k] = v
                            new_tokens[new_tokens.index(k)] = new_k
                            to_delete.append(k)
                            break
        for k in to_delete:
            del ent2content[k]

        for k, v in mem_num:
            if k not in head_num and k != "tmp_input":
                head_num.append(k)

        for k, v in mem_str:
            if k not in head_str:
                head_str.append(k)

        if any([_ in pos_tag for _ in ['RBR', 'RBS', 'JJR', 'JJS']]) and len(head_num) == 0:
            for col_idx, k in zip(range(len(cols) - 1, -1, -1), cols[::-1]):
                if mapping[col_idx] == 'num':
                    head_num.append(k)
                    break

        return " ".join(new_tokens), mem_str, mem_num, head_str, head_num, ent2content

    def run(self, table_name, sent, masked_sent, pos_tag, mem_str, mem_num, head_str, head_num):
        t = pandas.read_csv(os.path.join(self.folder, table_name), delimiter="#", encoding='utf-8')
        t.fillna('')
        res = dynamic_programming(table_name, t, sent, masked_sent, pos_tag, mem_str,
                                  mem_num, head_str, head_num, 7)
        return res[-1]

    def normalize(self, sent):
        sent = sent.lower()
        sent = sent.replace(',', '')
        sent = replace_number(sent)
        sent, _, pos_tags = self.get_lemmatize(sent, True)
        return sent, pos_tags

    def parse(self, table_name, sent, debug=False):
        sent, pos_tags = self.normalize(sent)
        raw_sent = " ".join(sent)
        linked_sent, pos = self.entity_link(table_name, sent, pos_tags)
        masked_sent, mem_str, mem_num, head_str, head_num, mapping = self.initialize_buffer(table_name, linked_sent, pos, raw_sent)

        if debug:
            print("Input to dynmiac programmer: ", masked_sent, mem_str, mem_num, head_str, head_num)
        
        result = self.run(table_name, raw_sent, masked_sent, pos, mem_str, mem_num, head_str, head_num)

        return result, masked_sent, mapping
    
    def hash_string(self, string):
        import hashlib
        sha = hashlib.sha256()
        sha.update(string.encode())
        return sha.hexdigest()[:16]


    def distribute_parse(self, inputs):
        table_name, sent = inputs
        hash_id = self.hash_string(sent)
        if not os.path.exists('tmp/results/{}.json'.format(hash_id)):
            sent, pos_tags = self.normalize(sent)
            raw_sent = " ".join(sent)
            linked_sent, pos = self.entity_link(table_name, sent, pos_tags)
            masked_sent, mem_str, mem_num, head_str, head_num, mapping = self.initialize_buffer(table_name, linked_sent, pos, raw_sent)
            
            result = self.run(table_name, raw_sent, masked_sent, pos, mem_str, mem_num, head_str, head_num)

            with open('tmp/results/{}.json'.format(hash_id), 'w') as f:
                json.dump((inputs[1], result, self.title_mapping[table_name]), f, indent=2)

            return inputs[1], result, self.title_mapping[table_name]
        else:
            with open('tmp/results/{}.json'.format(hash_id), 'r') as f:
                data = json.load(f)
            return data

    def preprocess(self, inputs):
        table_name, sent = inputs
        sent, pos_tags = self.normalize(sent)
        raw_sent = " ".join(sent)
        linked_sent, pos = self.entity_link(table_name, sent, pos_tags)
        masked_sent, mem_str, mem_num, head_str, head_num, mapping = self.initialize_buffer(table_name, linked_sent, pos, raw_sent)

        columns = self.get_table(table_name).columns.to_list()
        indexed_columns = []
        for m in head_str + head_num:
            if m in columns:
                indexed_columns.append(columns.index(m))
        if len(indexed_columns) == 0:
            indexed_columns = [0, 1, 2]
            assert len(columns) >= 3, "too few columns"

        return indexed_columns, masked_sent