import math
import re
from copy import deepcopy
from typing import List

def count_parameters(model):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    millnames = ['',' K',' M',' B']
    millidx = max(0, min(len(millnames)-1, int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
    return '{:.3f}{}'.format(n / 10**(3 * millidx), millnames[millidx])


def calculate_acc(gold_eqns, pred_eqns, nums):
    correct, total = 0, 0
    for i in range(len(gold_eqns)):
        gold_eqn, pred_eqn = gold_eqns[i], pred_eqns[i]
        gold_eqn = ''.join(gold_eqn)
        pred_eqn = ''.join(pred_eqn)

        mmap = dict()
        for j in range(len(nums[i])):
            mmap[f'number{j}'] = float(nums[i][j])

        for key, val in mmap.items():
            gold_eqn = re.sub(key, str(val), gold_eqn)
            pred_eqn = re.sub(key, str(val), pred_eqn)
        gold_ans = eval(gold_eqn)
        try:
            pred_ans = eval(pred_eqn)
            if abs(pred_ans - gold_ans) <= 1e-4:
                correct += 1
            total += 1
        except:
            total += 1
    return correct, total


def calculate_acc_code(gold_eqns, pred_eqns):
    correct, total = 0, 0
    for i in range(len(gold_eqns)):
        gold_code, pred_code = gold_eqns[i], pred_eqns[i]
        gold_code = ''.join(gold_code)
        gold_code = re.sub(r'<SEP>', '\n', gold_code)
        exec(gold_code)
        gold_res = float(locals()['RES_'])

        pred_code = ''.join(pred_code)
        pred_code = re.sub(r'<SEP>', '\n', pred_code)
        try:
            exec(pred_code)
            exec_res = float(locals()['RES_'])
            if abs(exec_res - gold_res) <= 0.01:
                correct += 1
        except:
            pass
        total += 1
    return correct, total


def from_prefix_to_infix(expression):
    """
    convert prefix equation to infix equation
        Args:
            expression (list): prefix expression.    
        Returns:
            (list): infix expression.
    """
    st = list()
    last_op = []
    priority = {"<BRG>": 0, "=": 1, "+": 2, "-": 2, "*": 3, "/": 3, "^": 4}
    expression = deepcopy(expression)
    expression.reverse()
    for symbol in expression:
        if symbol not in ['+', '-', '*', '/', '^', "=", "<BRG>"]:
            st.append([symbol])
        else:
            n_left = st.pop()
            n_right = st.pop()
            left_first = False
            right_first = False
            if len(n_left) > 1 and priority[last_op.pop()] < priority[symbol]:
                left_first = True
            if len(n_right) > 1 and priority[last_op.pop()] <= priority[symbol]:
                right_first = True
            if left_first:
                n_left = ['('] + n_left + [')']
            if right_first:
                n_right = ['('] + n_right + [')']
            st.append(n_left + [symbol] + n_right)
            last_op.append(symbol)
    res = st.pop()
    return res


def from_prefix_to_postfix(expression):
    r"""convert prefix equation to postfix equation

    Args:
        expression (list): prefix expression.
    
    Returns:
        (list): postfix expression.
    """
    st = list()
    expression = deepcopy(expression)
    expression.reverse()
    for symbol in expression:
        if symbol not in ['+', '-', '*', '/', '^', "=", "<BRG>"]:
            st.append([symbol])
        else:
            n1 = st.pop()
            n2 = st.pop()
            st.append(n1 + n2 + [symbol])
    res = st.pop()
    return res


def from_prefix_to_code(prefix_expr):
    postfix_expr = from_prefix_to_postfix(prefix_expr)
    step_count = 1
    code_expr, var_ls = list(), list()
    for token in postfix_expr:
        if token not in ['+', '-', '*', '/']:
            assert re.match(r'^number\d$', token) or re.match(r'\d+\.?\d*', token)
                # must be numberX or constant
            var_ls.append(token)
        else:
            operator = token
            assert len(var_ls) >= 2
            right_operand, left_operand = var_ls.pop(), var_ls.pop()
            var_ls.append(f'm_{step_count}')

            """ variable assignment"""
            if left_operand.startswith('number'):
                left_var_name = re.sub(r'number', 'NUM', left_operand)
                code_expr.extend([left_var_name, '=', left_operand, '<SEP>'])
            else:
                left_var_name = left_operand

            if right_operand.startswith('number'):
                right_var_name = re.sub(r'number', 'NUM', right_operand)
                code_expr.extend([right_var_name, '=', right_operand, '<SEP>'])
            else:
                right_var_name = right_operand
            
            """ calculation """
            code_expr.extend([f'm_{step_count}', '=', left_var_name, operator, right_var_name, '<SEP>'])
            step_count += 1
    code_expr[-6] = 'RES_'
    code_expr = code_expr[:-1]
    return code_expr


def from_prefix_to_deductive(prefix_expr):
    """ transfer prefix expression into DeductReasoner format"""
    postfix_expr = from_prefix_to_postfix(prefix_expr)
    step_count = 1
    code_expr, var_ls = list(), list()
    for token in postfix_expr:
        if token not in ['+', '-', '*', '/']:
            assert re.match(r'^number\d$', token) or re.match(r'\d+\.?\d*', token)
                # must be numberX or constant
            var_ls.append(token)
        else:
            operator = token
            assert len(var_ls) >= 2
            right_operand, left_operand = var_ls.pop(), var_ls.pop()
            var_ls.append(f'm_{step_count}')
            
            l, r = left_operand, right_operand
            assert l.startswith('m_') or l.startswith('number') or re.match(r'\d+\.?\d*', l)
            assert r.startswith('m_') or r.startswith('number') or re.match(r'\d+\.?\d*', r)

            # case 1, 2, 3
            if l.startswith('m_') and r.startswith('m_') \
                and int(l[-1]) > int(r[-1]):
                left_operand, right_operand = r, l
                if operator in ['-', '/']:
                    operator += '_rev'

            # case 4, 6
            elif (l.startswith('number') or re.match(r'\d+\.?\d*', l)) \
                and r.startswith('m_'):
                left_operand, right_operand = r, l
                if operator in ['-', '/']:
                    operator += '_rev'

            # case 5
            elif l.startswith('number') and r.startswith('number') \
                and int(l[-1]) > int(r[-1]):
                left_operand, right_operand = r, l
                if operator in ['-', '/']:
                    operator += '_rev'

            # case 7, 8, 9
            elif re.match(r'\d+\.?\d*', l) and r.startswith('m_'):
                left_operand, right_operand = r, l
                if operator in ['-', '/']:
                    operator += '_rev'

            code_expr.extend([left_operand, right_operand, operator])
            step_count += 1
    return code_expr
            

def compute(left: float, right:float, op:str):
    if op == "+":
        return left + right
    elif op == "-":
        return left - right
    elif op == "*":
        return left * right
    elif op == "/":
        return (left * 1.0 / right) if right != 0 else  (left * 1.0 / 0.001)
    elif op == "-_rev":
        try:
            return right - left
        except:
            print(left, right)
            exit()
    elif op == "/_rev":
        return (right * 1.0 / left) if left != 0 else  (right * 1.0 / 0.001)
    elif op == "^":
        try:
            return math.pow(left, right)
        except:
            return 0
    elif op == "^_rev":
        try:
            return math.pow(right, left)
        except:
            return 0
    else:
        raise NotImplementedError(f"not implementad for op: {op}")


def compute_value(equations, num_list, num_constant, uni_labels, constant_values: List[float] = None):
    current_value = 0
    for equation in equations:
        left_var_idx, right_var_idx, op_idx, _ = equation
        left_number = num_list[left_var_idx - num_constant] if left_var_idx >= num_constant else None
        if left_var_idx != -1 and left_var_idx < num_constant: ## means left number is a
            left_number = constant_values[left_var_idx]
        right_number = num_list[right_var_idx - num_constant] if right_var_idx >= num_constant else constant_values[right_var_idx]
        op = uni_labels[op_idx]
        if left_number is None:
            assert current_value is not None
            current_value = compute(current_value, right_number, op)
        else:
            current_value = compute(left_number, right_number, op)
    return current_value


def compute_value_for_incremental_equations(equations, num_list, num_constant, uni_labels, constant_values: List[float] = None):
    current_value = 0
    store_values = []
    grounded_equations = []
    for eq_idx, equation in enumerate(equations):
        left_var_idx, right_var_idx, op_idx, _ = equation
        assert left_var_idx >= 0
        assert right_var_idx >= 0
        if left_var_idx >= eq_idx and left_var_idx < eq_idx + num_constant:  ## means
            left_number = constant_values[left_var_idx - eq_idx]
        elif left_var_idx >= eq_idx + num_constant:
            left_number = num_list[left_var_idx - num_constant - eq_idx]
        else:
            assert left_var_idx < eq_idx  ## means m
            m_idx = eq_idx - left_var_idx
            left_number = store_values[m_idx - 1]

        if right_var_idx >= eq_idx and right_var_idx < eq_idx + num_constant:## means
            right_number = constant_values[right_var_idx- eq_idx]
        elif right_var_idx >= eq_idx + num_constant:
            right_number = num_list[right_var_idx - num_constant - eq_idx]
        else:
            assert right_var_idx < eq_idx ## means m
            m_idx = eq_idx - right_var_idx
            right_number = store_values[m_idx - 1]

        op = uni_labels[op_idx]
        current_value = compute(left_number, right_number, op)
        grounded_equations.append([left_number, right_number, op, current_value])
        store_values.append(current_value)
    return current_value, grounded_equations


def compute_value_for_parallel_equations(parallel_equations:List[List], num_list, num_constant, uni_labels, constant_values: List[float] = None):
    current_value = 0
    store_values = []
    grounded_equations = []
    accumulate_eqs = [0]
    for p_idx, equations in enumerate(parallel_equations):
        current_store_values = []
        for eq_idx, equation in enumerate(equations):
            left_var_idx, right_var_idx, op_idx, _ = equation
            assert left_var_idx >= 0
            assert right_var_idx >= 0
            if left_var_idx >= accumulate_eqs[p_idx] and left_var_idx < accumulate_eqs[p_idx] + num_constant:  ## means
                left_number = constant_values[left_var_idx - accumulate_eqs[p_idx]]
            elif left_var_idx >= accumulate_eqs[p_idx] + num_constant:
                left_number = num_list[left_var_idx - num_constant - accumulate_eqs[p_idx]]
            else:
                assert left_var_idx < accumulate_eqs[p_idx]  ## means m
                m_idx = accumulate_eqs[p_idx] - left_var_idx
                left_number = store_values[left_var_idx]

            if right_var_idx >= accumulate_eqs[p_idx] and right_var_idx < accumulate_eqs[p_idx] + num_constant:## means
                right_number = constant_values[right_var_idx- accumulate_eqs[p_idx]]
            elif right_var_idx >= accumulate_eqs[p_idx] + num_constant:
                right_number = num_list[right_var_idx - num_constant - accumulate_eqs[p_idx]]
            else:
                assert right_var_idx < accumulate_eqs[p_idx] ## means m
                m_idx = accumulate_eqs[p_idx] - right_var_idx
                right_number = store_values[right_var_idx]

            op = uni_labels[op_idx]
            current_value = compute(left_number, right_number, op)
            grounded_equations.append([left_number, right_number, op, current_value])
            current_store_values.append(current_value)
        store_values = current_store_values + store_values
        accumulate_eqs.append(accumulate_eqs[len(accumulate_eqs) - 1] + len(equations))
    return current_value, grounded_equations


def is_value_correct(predictions, labels, num_list, num_constant, uni_labels, constant_values: List[float] = None, consider_multiple_m0=False, use_parallel_equations: bool = False):
    pred_grounded_equations = None
    gold_grounded_equations = None
    if consider_multiple_m0:
        if use_parallel_equations:
            pred_val, pred_grounded_equations = compute_value_for_parallel_equations(predictions, num_list, num_constant, uni_labels, constant_values)
        else:
            pred_val, pred_grounded_equations = compute_value_for_incremental_equations(predictions, num_list, num_constant, uni_labels, constant_values)
    else:
        pred_val = compute_value(predictions, num_list, num_constant, uni_labels, constant_values)
    if consider_multiple_m0:
        if use_parallel_equations:
            gold_val, gold_grounded_equations = compute_value_for_parallel_equations(labels, num_list, num_constant, uni_labels, constant_values)
        else:
            gold_val, gold_grounded_equations = compute_value_for_incremental_equations(labels, num_list, num_constant, uni_labels,  constant_values)
    else:
        gold_val = compute_value(labels, num_list, num_constant, uni_labels, constant_values)
    if math.fabs((gold_val- pred_val)) < 1e-4:
        return True, pred_val, gold_val, pred_grounded_equations, gold_grounded_equations
    else:
        return False, pred_val, gold_val, pred_grounded_equations, gold_grounded_equations
