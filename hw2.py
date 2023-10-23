import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X = [0] * 26
    with open(filename, encoding='utf-8') as f:
        for line in f:
            for char in line:
                char = char.lower()
                dist = ord(char) - ord('a')
                if 0 <= dist < 26:
                    X[dist] = X[dist] + 1

    return X

# You are free to implement it as you wish!
# Happy Coding!

# Print “Q1” followed by the 26 character counts for letter.txt
def Q1(X):
    print('Q1')
    for i in range(26):
        print(chr(ord('A')+i), end ="")
        print("", X[i])

# Compute X1 log e1 and X1 log s1
def Q2(X, p_list):
    print('Q2')
    e, s = p_list
    result1 = X[0] * math.log(e[0])
    result2 = X[0] * math.log(s[0])
    print('%.4f' % result1)
    print('%.4f' % result2)

# Compute F(English) and F(Spanish)
def Q3(X, p_list):
    print('Q3')
    p_english = 0.6
    p_spanish = 0.4
    tempeng = 0
    tempspan = 0
    for i in range(26):
        tempeng = tempeng + X[i] * math.log(p_list[0][i])
    for i in range(26):
        tempspan = tempspan + X[i] * math.log(p_list[1][i])
    english = math.log(p_english) + tempeng
    spanish = math.log(p_spanish) + tempspan
    print('%.4f' % english)
    print('%.4f' % spanish)
    return english, spanish

# Compute P(Y = English | X)
def Q4(F_func):
    print('Q4')
    english, spanish = F_func
    diff = spanish - english
    if diff >= 100:
        result = 0
    elif diff <= -100:
        result = 1
    else:
        result = 1 / (1 + math.e**diff)
    print('%.4f' % result)

if __name__ == '__main__':
    X = shred('letter.txt')
    p_list = get_parameter_vectors()
    Q1(X)
    Q2(X, p_list)
    F_func = Q3(X, p_list)
    Q4(F_func)