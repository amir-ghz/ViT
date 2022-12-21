# binary and bit flip conversions


def int_to_binary(integer, b):
    bin_list = []
    is_pos = True
    if integer < 0:
        is_pos = False
        integer = -integer

    binary_string = ''
    while(integer > 0):
        digit = integer % 2
        binary_string += str(digit)
        integer = integer // 2
    binary_string = binary_string[::-1]

    for i in binary_string:
        bin_list.append(i)


    my_list = bin_list[::-1]

    for i in range(0, (b -1 - len(bin_list))):
        my_list.append('0')

    if is_pos:
        my_list.append('0')
    else:
        my_list.append('1')

    my_list = my_list[::-1]

    return my_list


def bit_flip(val, pos):

    positioner = len(val)-1

    if val[positioner-pos] == '0':
        val[positioner-pos] = '1'
    #else:
        #val[positioner-pos] = '0'
    return val


def binary_to_int(val):
    result = 0
    for i in (range(1, len(val))):
        if(val[i] == '1'):
            result = result + 2**(len(val)-i-1)
    if(val[0] == '1'):
        result = result*-1

    return result


        
