import time
import random
import matplotlib.pyplot as plt
import numpy as np

def LoveThyNeighbor(L):
    dp = [0] * (len(L)+1)
    dp[1] = L[0]
    dp[2] = max(L[0], L[1])
    for i in range(3, len(L)+1):
        dp[i] = max(dp[i-1], dp[i-2] + L[i-1])
    return dp

def MVBLN(L, k):
    n = len(L)
    dp = [[0] * (k + 1) for _ in range(n+1)]

    for j in range(k + 1):
        dp[1][j] = L[0]

    baseDp = LoveThyNeighbor(L)
    for i in range(n+1):
        dp[i][0] = baseDp[i]

    for i in range(2, n+1):
        for j in range(1, k + 1):
            dp[i][j] = max(dp[i-1][j], dp[i-1][j-1] + L[i-1], dp[i-2][j] + L[i-1])

    mvbln_value = dp[n][k]

    b = [0] * n
    i, j = n , k
    while i > 0 and j >= 0:
        if i >= 2:
            if dp[i][j] == dp[i-2][j] + L[i-1]:
                b[i-1] = 1
                i -= 2
            elif dp[i][j] == dp[i-1][j-1] + L[i-1]:
                b[i-1] = 1
                i -= 1
                j -= 1
            else:
                i-=1
        else:
            if dp[i][j] == dp[i-1][j-1] + L[i-1]:
                b[i-1] = 1
                i -= 1
                j -= 1
            else:
                i-=1

    while i > 0:
        if baseDp[i]==baseDp[i-1]:
            i-=1
        elif i>=2 and baseDp[i]==baseDp[i-2]+L[i-1]:
            b[i-1]=1
            i-=2
        else:
            i-=1

    return b, mvbln_value

def correctness_test():    
    L1 = [100, 300, 400, 500, 400]
    k1 = 2
    L2 = [10, 100, 300, 400, 50, 4500, 200, 30, 90]
    k2 = 2
    L3 = [100, 300, 400, 50]
    k3 = 1
    L4 = np.random.randint(10,30,10)
    k4 = 3

    b1, mvbln1 = MVBLN(L1, k1)
    print(b1)  #  [1, 0, 1, 1, 1]
    print(mvbln1)  #  1400

    b2, mvbln2 = MVBLN(L2, k2)
    print(b2)  #  [1, 0, 1, 1, 0, 1, 1, 0, 1]
    print(mvbln2)  #  5500

    b3, mvbln3 = MVBLN(L3, k3)
    print(b3)  #  [0, 1, 1, 0]
    print(mvbln3)  #  700

    b4, mvbln4 = MVBLN(L4, k4)
    print(b4)  
    print(L4)  
    print(mvbln4)  

def performance_test(L,N,n,k):
    start = time.time()
    b, mvbln = MVBLN(L, k)
    end = time.time()
    return end-start

def plot_2d_graph(n, z1, z2):
    plt.plot(n, z1, 'r', label='Experimental Result')
    plt.plot(n, z2, 'b', label='Thoeretical Result')
    plt.xlabel('nk')
    plt.ylabel('Running Time')
    plt.legend()
    plt.show()

def run_and_plot():
    N = 10000
    n = [i for i in range(1000, 10000, 500)]
    k = [100]
    z=[]
       
    for i in n:
        for j in k:
            temp=[]
            for _ in range(10):
                L = [ele for ele in random.sample(range(N), i)]
                temp.append(performance_test(L, N, i, j))
            z.append(sum(temp)/10)
    print(n)
    #print(z)
    for i in z:
        print(i)
    ratio = 0.00000063
    plot_2d_graph([j*100 for j in n],z,[i*100*ratio for i in n])


if __name__ == '__main__':
    correctness_test()
    run_and_plot()
    



    
        

