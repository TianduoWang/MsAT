division = {
    'number0': lambda x: x[0],
    '+ number0 number1': lambda x: x[0] + x[1],
    '- number0 number1': lambda x: x[0] - x[1],
    '* number0 number1': lambda x: x[0] * x[1],
    '/ number0 number1': lambda x: x[0] / x[1],
    '+ + number0 number1 number2': lambda x: x[0] + x[1] + x[2],
    '+ - number0 number1 number2': lambda x: x[0] - x[1] + x[2],
    '+ * number0 number1 number2': lambda x: x[0] * x[1] + x[2],
    '+ / number0 number1 number2': lambda x: x[0] / x[1] + x[2],
    '- + number0 number1 number2': lambda x: x[0] + x[1] - x[2],
    '- - number0 number1 number2': lambda x: x[0] - x[1] - x[2],
    '- * number0 number1 number2': lambda x: x[0] * x[1] - x[2],
    '- / number0 number1 number2': lambda x: x[0] / x[1] - x[2],
    '* + number0 number1 number2': lambda x: (x[0] + x[1]) * x[2],
    '* - number0 number1 number2': lambda x: (x[0] - x[1]) * x[2],
    '* * number0 number1 number2': lambda x: x[0] * x[1] * x[2],
    '* / number0 number1 number2': lambda x: x[0] / x[1] * x[2],
    '/ + number0 number1 number2': lambda x: (x[0] + x[1]) / x[2],
    '/ - number0 number1 number2': lambda x: (x[0] - x[1]) / x[2],
    '/ * number0 number1 number2': lambda x: x[0] * x[1] / x[2],
    '/ / number0 number1 number2': lambda x: x[0] / x[1] / x[2],
}



