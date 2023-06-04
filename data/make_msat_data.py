import random
import argparse
import json
import re
from copy import deepcopy
from collections import OrderedDict, Counter
import itertools
import pandas as pd
from tqdm import tqdm
import primefac
import numpy as np
import os
import sys
sys.path.append('.')
from cvxopt import matrix, solvers

from core.utils import from_prefix_to_infix
from data.eqn_templates import division


def find_prime_factors(num, list=None):
    if list is None:
        list = []
    for i in range(2,num):
        while num % i == 0:
            list.append(i)
            num = int(num / i)
            if num > 1:
                find_prime_factors(num)
    return list


def left_transform(tree_op, tree_right_child, out):
    op_trans = {'+': '-', '-': '+', '*': '/', '/': '*'}
    new_op = op_trans[tree_op]
    new_out = [new_op, *out, *tree_right_child]
    return new_out


def right_transform(tree_op, tree_left_child, out):
    op_trans = {'+': '-', '-': '-', '*': '/', '/': '/'}
    new_op = op_trans[tree_op]
    if tree_op in ['-', '/']:
        new_out = [new_op, *tree_left_child, *out]
    else:
        new_out = [new_op, *out, *tree_left_child]
    return new_out


def process_mask_eq(prefix_eqn_str, question_obj, last_var):
    prefix_eq = prefix_eqn_str.split()
    out = [last_var]
    while len(prefix_eq) > 1:
        tree_op = prefix_eq[0]

        if len(prefix_eq) == 3:
            split_point = 2
        else:
            assert len(prefix_eq) > 3 and (len(prefix_eq)-1)%2 == 0
            state = 0
            seen_num = 0
            for ptr in range(1, len(prefix_eq)):
                if prefix_eq[ptr] in ['+', '-', '*', '/']:
                    state += 1
                else:
                    seen_num += 1
                    if seen_num > 1:
                        state -= 1
                    if state == 0:
                        split_point = ptr + 1
                        break
        
        tree_left_child = prefix_eq[1:split_point]
        tree_right_child = prefix_eq[split_point:]

        if question_obj in tree_left_child:
            out = left_transform(tree_op, tree_right_child, out)
            prefix_eq = deepcopy(tree_left_child)
        else:
            assert question_obj in tree_right_child
            out = right_transform(tree_op, tree_left_child, out)
            prefix_eq = deepcopy(tree_right_child)
    return ' '.join(out)


def make_one_inst(eqn_template, num_list, ques_var_idx):

    """ make sure nums are at most 3-digit after period """
    for num in num_list:
        try:
            assert abs(num - round(num, 3)) < 1e-5
        except:
            print(eqn_template)
            print(num_list)
            exit('num not right')

    num_nece_vars = (len(eqn_template.split())+1)//2
    nece_var_ls = []
    for i in range(num_nece_vars):
        nece_var_ls.append(f'number{i}')

    for var in nece_var_ls:
        """ to ensure all necessary vars appear in equation """
        assert var in eqn_template

    """ add ans var in necessary var list """
    nece_var_ls.append(f'number{len(num_list)-1}')

    """ include both necessary vars and unused vars """
    all_vars_ls = []
    for i in range(len(num_list)):
        all_vars_ls.append(f'number{i}')

    """ make question-names list """
    possible_names = []
    for offset in range(1, 26):
        possible_names.append(chr(65+offset))

    ques_names = random.sample(possible_names, len(all_vars_ls))
    
    random.shuffle(ques_names)
    assert len(ques_names) == len(all_vars_ls)

    """ make var to question-name mapping """
    var_ques_name_map = dict()
    for sym_idx in range(len(all_vars_ls)):
        var_ques_name_map[f'number{sym_idx}'] = ques_names[sym_idx]

    infix_expr_tokens = from_prefix_to_infix(eqn_template.split())
    infix_expr_tokens.extend(['=', nece_var_ls[-1]])

    """ select which var to ask (must be necessary var) """
    question_var = nece_var_ls[ques_var_idx]

    """ build body and context """
    body, context = list(), list()
    for idx, token in enumerate(infix_expr_tokens):
        if token in ['+', '-', '*', '/', '=', '(', ')']:
            body.append(token)
        elif token == question_var:
            body.append(var_ques_name_map[token])
        else:
            if random.random() < 0.8: 
                # 0.6 的几率用变量名
                body.append(var_ques_name_map[token])
                context.append((var_ques_name_map[token], token))
            else: 
                # 0.4 的几率选真实数字
                body.append(token)

    """ add unused var in context """
    for var in all_vars_ls:
        if var not in infix_expr_tokens:
            context.append((var_ques_name_map[var], var))
    random.shuffle(context)

    """ concat context, body, and question """
    input_question = ''
    for ques_name, var in context:
        input_question += f'{ques_name} = {var} . '
    input_question += ' '.join(body)
    input_question += f' . {var_ques_name_map[question_var]} ?'

    """ make sure all var appear in question"""
    for var in all_vars_ls:
        if var != question_var:
            assert var in input_question

    """ change var order """
    old_to_new_var_map = {}
    var_cnt = 0
    for token in input_question.split():
        if re.match(r'^number\d$', token):
            old_to_new_var_map[token] = f'NUM{var_cnt}'
            var_cnt += 1

    try:
        assert var_cnt == len(all_vars_ls) - 1
    except:
        print(input_question)
        print(all_vars_ls)
        print(old_to_new_var_map)
        exit()

    for old, new in old_to_new_var_map.items():
        input_question = re.sub(old, new, input_question)
        eqn_template = re.sub(old, new, eqn_template)
    
    """ build output """
    if question_var == nece_var_ls[-1]:
        prefix_expr = eqn_template
    else:
        new_last_var = old_to_new_var_map[nece_var_ls[-1]]
        prefix_expr = process_mask_eq(eqn_template, question_var, new_last_var)

    """ change num_list"""
    new_num_list = []
    for old_var in old_to_new_var_map:
        old_var_idx = int(old_var[-1])
        new_num_list.append(num_list[old_var_idx])
    assert len(new_num_list) == len(num_list) - 1 # without question-obj's num

    """ transfer NUM to number """
    for i in range(10):
        input_question = input_question.replace('NUM', 'number')
        prefix_expr = prefix_expr.replace('NUM', 'number')

    """ find numerical answer"""
    old_question_var_idx = int(question_var[-1])
    ans = str(num_list[old_question_var_idx])

    """ build Numbers from new_num_list"""
    num_list_str = ' '.join(str(num) for num in new_num_list)

    """ verify """
    new_infix = from_prefix_to_infix(prefix_expr.split())
    new_infix = ' '.join(new_infix)
    for i, num_str in enumerate(num_list_str.split()):
        new_infix =re.sub(f'number{i}', num_str, new_infix)

    try:
        comp_result = eval(new_infix)
    except:
        return None
    assert abs(comp_result - float(ans)) < 1e-5

    return {'Question': input_question, 'Equation': prefix_expr, 'Answer': ans, "Numbers": num_list_str}


def eval_infix(infix_expr, num_list):
    copy_infix_expr = infix_expr
    for i in range(len(num_list)):
        copy_infix_expr = re.sub(f'number{i}', str(num_list[i]), copy_infix_expr)
    comp_result = eval(copy_infix_expr)
    return comp_result


def prepare_num_list(prefix_expr, sample_range):
    """Given an equation template, 
           return number list and add noise variables
    Args:
        prefix_expr (str)
    """

    infix_expr = ' '.join(from_prefix_to_infix(prefix_expr.split()))

    num_vars = (len(prefix_expr.split())+1)//2
    assert len(sample_range) == num_vars
    comp_step = num_vars - 1

    step, max_trial_steps = 0, 50
    found = False
    while step < max_trial_steps:
        num_list = []
        failed = False
        for i in range(num_vars):
            if isinstance(sample_range[i], int):
                random_int = int(random.random() * sample_range[i])
                random_int = max(2, random_int)
            else:
                res = int(sample_range[i](num_list))
                factors = list(primefac.primefac(res))
                num_factor = len(factors)
                if num_factor < 2:
                    failed = True
                    break
                sample_num = random.choice(list(range(1, num_factor)))
                samples = random.sample(factors, sample_num)
                random_int = 1
                for s in samples:
                    random_int *= s

                if random_int > 9999:
                    failed = True
                    break

            num_list.append(random_int)
        if failed:
            continue
        comp_result = eval_infix(infix_expr, num_list)
        step += 1
        if comp_result>0 and comp_result<9999 and abs(int(comp_result) - comp_result)<1e-4:
            found = True
            break

    if not found:
        return

    """ add expression results """
    num_list.append(comp_result)
    assert len(num_list) <= 10
    return num_list


def add_unused_vars(eqn_template, num_list):
    comp_step = (len(eqn_template.split())-1)//2
    out_num_list = num_list[:-1]
    comp_result = num_list[-1]
    """ transform int to float randomly """

    """ add unused variables """
    unused_prob = 0.7
    params = {
        1: (unused_prob, [1, 2, 3, 4]),
        2: (unused_prob, [1, 2, 3]),
        3: (unused_prob, [1, 2]),
        }
    
    prob_unused = params[comp_step][0]
    num_unused_dist = params[comp_step][1]

    if random.random() < prob_unused:
        num_unused_vars = random.choice(num_unused_dist)
        assert len(out_num_list) + num_unused_vars <= 9
        for i in range(num_unused_vars):
            n = random.choice(list(range(1, 1000)))
            out_num_list.append(n)
    out_num_list.append(comp_result)
    return out_num_list


def generate_step_split(total_num, difficulty):

    P = matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    q = matrix([0., 30., 0.])
    G = matrix([[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    h = matrix([0., 0., 0.])
    A = matrix([[1., 1.], [1., 2.], [1., 3.]])
    b = matrix([total_num, total_num * difficulty])
    result = solvers.qp(P,q,G,h,A,b)

    output = []
    for i in result['x']:
        output.append(int(round(i)))
    return output


def get_template(operator_prefix, range_ls=[100, 400, 2000]):
    num_ops, num_oprands = len(operator_prefix), len(operator_prefix)+1
    prefix = ' '.join(operator_prefix)
    suffix = ' '.join(f'number{j}' for j in range(num_oprands))
    template = prefix + ' ' + suffix

    range_list = range_ls

    values = []
    t = list(range(len(range_ls)))
    values.append(random.choice(t))
    for j in range(num_ops):
        op, oprand = prefix.split()[-(j+1)], suffix.split()[j+1]

        if op == '+':
            values.append(random.choice(t))
        elif op == '-':
            if isinstance(values[-1], int):
                values.append(random.choice(range(values[-1]+1)))
            else:
                values.append(random.choice(range(values[0]+1)))
        elif op == '*':
            values.append(0)
        elif op == '/':
            key = ' '.join(template.split()[num_ops - j : num_ops + j + 1])
            values.append(division[key])
        else:
            raise NotImplementedError('check get_template()')

    v = []
    for item in values:
        if isinstance(item, int):
            v.append(range_list[item])
        else:
            v.append(item)

    return template, v
    


if __name__ == '__main__':
    
    random.seed(2)

    config_parser = argparse.ArgumentParser()
    config_parser.add_argument('--dataset_name', default='msat_trial', type=str)
    config_parser.add_argument('--total_num', default=85000, type=int)
    config_parser.add_argument('--train_num', default=80000, type=int)
    config_parser.add_argument('--difficulty', default=2.4, type=float)
    config = config_parser.parse_args()

    dataset_name = config.dataset_name
    total_num, train_num, train_difficulty = config.total_num, config.train_num, config.difficulty

    step_split = generate_step_split(total_num, train_difficulty)
    print(f'\nStep split: {step_split}')
    ops = ['+', '-', '*', '/']

    msat_data = []
    for i, step_cnt in enumerate(step_split):
        step = i + 1
        op_pmt = [p for p in itertools.product(ops, repeat=step)]

        num_pmt = len(op_pmt)

        cnt = 0
        temp = 0
        while cnt < step_cnt:
            operators = op_pmt[temp % num_pmt]
            tplt, value_range = get_template(operators)
            temp += 1
            nece_num_list = prepare_num_list(tplt, value_range)
            if nece_num_list:
                assert len(nece_num_list) == (len(tplt.split())+1)//2+1
                
                num_list = add_unused_vars(tplt, nece_num_list)

                nece_var_ls = list(range(len(nece_num_list)))
                assert len(nece_num_list) > 2
                ques_vars_idx = random.sample(nece_var_ls, len(nece_num_list)-2)
                for idx in ques_vars_idx:
                    inst = make_one_inst(tplt, num_list, idx)
                    if inst:
                        msat_data.append(inst)
                        cnt += 1

    random.shuffle(msat_data)
    msat_data = msat_data[:total_num]

    step_cnt, question_len = 0, 0
    for d in msat_data:
        step_cnt += (len(d['Equation'].split())-1)//2
        question_len += len(d['Question'].split())

    print(f'\nAvg step: {step_cnt / len(msat_data):.1f}')    
    print(f'Avg question len: {question_len / len(msat_data):.1f}')   


    train_msat = msat_data[:train_num]
    print(f'\nHave {len(train_msat)} training data.')


    test_msat = msat_data[train_num:]
    print(f'Have {len(test_msat)} test data.')

    
    if not os.path.exists(f'./data/{dataset_name}'):
        os.makedirs(f'./data/{dataset_name}')
    import csv 
    Details = ['Question', 'Equation', 'Answer', 'Numbers']
    with open(f'data/{dataset_name}/train.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(Details)
        for d in train_msat:
            write.writerow([d['Question'], d['Equation'], d['Answer'], d['Numbers']]) 

    Details = ['Question', 'Equation', 'Answer', 'Numbers']
    with open(f'data/{dataset_name}/dev.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(Details)
        for d in test_msat:
            write.writerow([d['Question'], d['Equation'], d['Answer'], d['Numbers']]) 
    print('\nFinished!')


