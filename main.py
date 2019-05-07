"""Módulo principal.
Lê os argumentos do input do usuário.
"""

import Matrix
import Print
import DigitClassifier
import numpy as np
import time
import os
import ctypes

def parte1():
    print("\nParte 1\n----------------------\nRot-givens\n")
    print("item a)")
    n = m = 64
    W = np.zeros((n, m))
    b = np.zeros((n, 1))
    for i in range(n):
        b[i] = 1
        W[i][i] = 2
        for j in range(m):
            if abs(i-j) == 1:
                W[i][j] = 1
            elif abs(i-j) > 1:
                W[i][j] = 0
    x_ref = np.linalg.solve(W, b)
    x = Matrix.solveLinear(W, n, m, b)

    max_err = Matrix.maxError(x_ref, x)
    print("erro absoluto máximo = %e" %max_err)
    print("solução de referência: ")
    print(np.transpose(x_ref))
    print("solução obtida: ")
    Print.okGreen(x)

    print("\nitem b)")
    n = 20
    m = 17
    W = np.zeros((n, m))
    b = np.zeros((n, 1))
    for i in range(n):
        b[i] = i+1
        for j in range(m):
            if abs(i-j) <= 4:
                W[i][j] = 1/(i+1+j)
            else:
                W[i][j] = 0
    x_ref = np.linalg.lstsq(W, b)[0]
    x = Matrix.solveLinear(W, n, m, b)

    max_err = Matrix.maxError(x_ref, x)
    print("erro absoluto máximo = %e" %max_err)
    print("solução de referência: ")
    print(x_ref[:,0])
    print("solução obtida: ")
    Print.okGreen(x)

def parte2():
    print("\nParte 2\n----------------------\nVários sistemas simultâneos\n")
    print("item a)")
    n = p= 64
    m = 3
    A = np.zeros((n, m))
    W = np.zeros((n, p))
    b1 = np.zeros((n, 1))
    b2 = np.zeros((n, 1))
    b3 = np.zeros((n, 1))

    for i in range(n):
        for j in range(p):
            if abs(i-j) == 1:
                W[i][j] = 1
            elif abs(i-j) > 1:
                W[i][j] = 0
            elif i==j:
                W[i][i] = 2
        for k in range(m):
            if k == 0:
                b1[i] = 1
                A[i][k] = 1
            elif k == 1:
                b2[i] = i+1
                A[i][k] = i+1
            elif k == 2:
                b3[i] = 2*i+1
                A[i][k] = 2*i+1

    x_ref1 = np.linalg.lstsq(W, b1)[0]
    x_ref2 = np.linalg.lstsq(W, b2)[0]
    x_ref3 = np.linalg.lstsq(W, b3)[0]

    x = Matrix.solveMultipleLinear(W, n, m, p, A)
    x1 = x[:,0]
    x2 = x[:,1]
    x3 = x[:,2]

    print("solução de referência (sistema 1): ")
    print(x_ref1[:,0])
    print("solução obtida (sistema 1): ")
    Print.okGreen(x1)
    print("\nsolução de referência (sistema 2): ")
    print(x_ref2[:,0])
    print("solução obtida (sistema 2): ")
    Print.okGreen(x2)
    print("\nsolução de referência (sistema 3): ")
    print(x_ref3[:,0])
    print("solução obtida (sistema 3): ")
    Print.okGreen(x3)

    max_err = Matrix.maxError(x_ref1, x1)
    print("\nerro absoluto máximo 1 = %e" %max_err)
    max_err = Matrix.maxError(x_ref2, x2)
    print("erro absoluto máximo 2 = %e" %max_err)
    max_err = Matrix.maxError(x_ref3, x3)
    print("erro absoluto máximo 3 = %e" %max_err)
    
    print("\nitem b)")
    n = 20
    p = 17
    m = 3
    A = np.zeros((n, m))
    W = np.zeros((n, p))
    b1 = np.zeros((n, 1))
    b2 = np.zeros((n, 1))
    b3 = np.zeros((n, 1))

    for i in range(n):
        for j in range(p):
            if abs(i-j) <= 4:
                W[i][j] = 1/(i+1+j)
            else:
                W[i][j] = 0
        for k in range(m):
            if k == 0:
                b1[i] = 1
                A[i][k] = 1
            elif k == 1:
                b2[i] = i+1
                A[i][k] = i+1
            elif k == 2:
                b3[i] = 2*i+1
                A[i][k] = 2*i+1

    x_ref1 = np.linalg.lstsq(W, b1)[0]
    x_ref2 = np.linalg.lstsq(W, b2)[0]
    x_ref3 = np.linalg.lstsq(W, b3)[0]

    x = Matrix.solveMultipleLinear(W, n, m, p, A)
    x1 = x[:,0]
    x2 = x[:,1]
    x3 = x[:,2]

    print("solução de referência (sistema 1): ")
    print(x_ref1[:,0])
    print("solução obtida (sistema 1): ")
    Print.okGreen(x1)
    print("\nsolução de referência (sistema 2): ")
    print(x_ref2[:,0])
    print("solução obtida (sistema 2): ")
    Print.okGreen(x2)
    print("\nsolução de referência (sistema 3): ")
    print(x_ref3[:,0])
    print("solução obtida (sistema 3): ")
    Print.okGreen(x3)

    max_err = Matrix.maxError(x_ref1, x1)
    print("\nerro absoluto máximo 1 = %e" %max_err)
    max_err = Matrix.maxError(x_ref2, x2)
    print("erro absoluto máximo 2 = %e" %max_err)
    max_err = Matrix.maxError(x_ref3, x3)
    print("erro absoluto máximo 3 = %e" %max_err)

def parte3():
    print("\nParte 3\n----------------------\nFatoração por matrizes não negativas\n")
    A = [[3/10, 3/5, 0], [1/2, 0, 1], [4/10, 4/5, 0]]
    n = m = 3
    p = 2
    W, H = Matrix.NMF(A, n, m, p)
    print("\nmatriz W:")
    Print.okGreen(W)
    print("matriz H:")
    Print.okGreen(H)

def parte4():
    Print.warning("*** Dependendo dos valores dos parâmetros, o tempo de execução pode ser muito alto. ***\n")
    parte4_teste1()
    voltar = False
    while not voltar:
        escolher = input("Deseja escolher valores para os parâmetros? (s/n): ")
        if escolher == "s":
            parte4_escolher()
        elif escolher == "n":
            voltar = True
        else:
            Print.fail("Opção inválida! Tente novamente.")

def parte4_teste1():
    ndig_treino = 18
    p = 10
    n_test = 1000
    parte4_rodar(ndig_treino, p, n_test)

def parte4_escolher():
    ndig_treino = int(input("Escolha um valor para ndig_treino: "))
    p = int(input("Escolha um valor para p: "))
    n_test = int(input("Escolha um valor para n_test: "))
    parte4_rodar(ndig_treino, p, n_test)
    
def parte4_rodar(ndig_treino, p, n_test):
    instance = DigitClassifier.DigitClassifier(p, ndig_treino, n_test)
    Print.okBlue("Teste: ndig_treino = %d p = %d n_test = %d\n----------------------" %(ndig_treino, p, n_test))
    Print.okBlue("Treinando os classificadores...")
    start_time = time.time()
    instance.train()
    Print.okBlue("Feito em %.3f segundos!"%(time.time() - start_time))
    Print.okBlue("Rodando os testes...")
    start_time = time.time()
    instance.test()
    Print.okBlue("Feito em %.3f segundos!"%(time.time() - start_time))
    hitRate, _, hitRatePerDigit = instance.results()
    Print.okGreen("Porcentual total de acertos: %.2f" %hitRate)
    Print.okGreen("Porcentual de acertos por dígito:")
    for i in range(10):
        Print.okGreen("%d: %.2f" %(i, hitRatePerDigit[i]))

def printOptions():
    Print.header("\n----------------------\n(1) Rot-givens\n(2) Vários sistemas simultâneos\n(3) Fatoração por matrizes não negativas\n(4) Classificação de dígitos")
    Print.header("(q) Sair")
    
def main():
    """Main function.
    """
    # enabling Virtual Terminal Processing for Windows to print colorful output!
    if isWindows():
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    Print.bold("\nBem vindo ao EP1 de xxxxxx!")
    exit = False
    while not exit:
        printOptions()
        item = input("\nEscolha uma parte do exercício: ")
        if item == "1":
            parte1()
        elif item == "2":
            parte2()
        elif item == "3":
            parte3()
        elif item == "4":
            parte4()
        elif item == "q":
            exit = True
            Print.okBlue("Fim do programa.")
        else:
            Print.fail("Opção inválida! Tente novamente.")

if __name__ == "__main__":
    main()
