import itertools
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns
import pulp

import warnings
warnings.filterwarnings("ignore")
n_customer=7
n_point=n_customer+1
vehicle_capacity=4
df=pd.DataFrame({
    'x':[20,16,31,37,11,32,4,23,20],
    'y':[0,29,27,30,29,19,9,22,0],
})

# get distance matrix

distances = pd.DataFrame(distance_matrix(df[['x', 'y']].values, df[['x', 'y']].values), index=df.index, columns=df.index).values

fig, ax = plt.subplots(figsize=(4,3))
sns.heatmap(distances, ax=ax, cmap='Blues', annot=True, fmt='.0f', cbar=True, cbar_kws={"shrink": .3}, linewidths=.1)
plt.title('distance matrix')
plt.show()

# check TSP state

plt.figure(figsize=(5, 5))

# draw problem state
for i, row in df.iterrows():
    if i == 0:
        plt.scatter(row['x'], row['y'], c='r')
        plt.text(row['x'] + 1, row['y'] + 1, 'depot')
    else:
        plt.scatter(row['x'], row['y'], c='black')
        plt.text(row['x'] + 1, row['y'] + 1, f'{i}')
        
plt.xlim([-10, 50])
plt.ylim([-10, 50])
plt.title('points: id')
plt.show()
""""
print(distances[1][2])
a=np.array([2,1])
b=np.array([3,6])
c=np.array([5,10])
d=np.array([7,1])
e=np.array([4,2])
depo=np.array([5,0])
x=np.array([2,3,5,7,4,5])
y=np.array([1,6,10,1,2,0])
plt.scatter(x,y)

L=[depo,a,b,c,d,e]

#距離の算出
for i in range(6):
    for j in range(6):
        if i != j:
            distance=[]
            distance=np.linalg.norm(L[i]-L[j])

"""

#定数の定義
M=10000
e=1
m=1
c=50
b=40
S_L=1/60
S_R=1/60
problem = pulp.LpProblem('KUPC2018_C', pulp.LpMinimize)

# 変数とその定義域の定義
x=pulp.LpVariable.dicts('x',((i,j) for i in range(n_point) for j in range(1,n_point+1)),lowBound=0,upBound=1,cat='Binary')
y=pulp.LpVariable.dicts('y',((i,j,k) for i in range(n_point) for j in range(1,n_point) for k in range(1,n_point+1)),lowBound=0,upBound=1,cat='Binary')
# we need to keep track of the order in the tour to eliminate the possibility of subtours
u = pulp.LpVariable.dicts('u', (i for i in range(n_point+1)), lowBound=1, upBound=n_point+1, cat='Integer')
t = pulp.LpVariable.dicts('t', (i for i in range(n_point+1)), lowBound=0, cat='Continuous')
a=pulp.LpVariable.dicts('a', (i for i in range(n_point+1)), lowBound=0,cat='Continuous')
p=pulp.LpVariable.dicts('p',((i,j) for i in range(n_point) for j in range(1,n_point)),lowBound=0,upBound=1,cat='Binary')
# set objective funct
problem+=t[n_point]

#t[n_point]==pulp.lpSum((distances[i][j])*(x[i,j])for i in range(n_point)for j in range(1,n_point+1))




problem += (pulp.lpSum(x[0,j]for j in range(1,n_point)))== 1


problem += (pulp.lpSum(x[j,n_point]for j in range(1,n_point))) == 1


#全ての頂点を回る

for i in range(1,n_point):
    for j in range(1,n_point):
        for k in range(1,n_point+1):
            if i !=j and j !=k and k != i:
                problem += (2*y[i,j,k])<= pulp.lpSum(x[h,i]for h in range(n_point)if h != i)+pulp.lpSum(x[l,k]for l in range(1,n_point)if l !=k)

for i in range(n_point):
    problem+=pulp.lpSum(y[i,j,k] for j in range(1,n_point) for k in range(1,n_point+1)if i != j and k != j and i != k) <= 1

for k in range(1,n_point+1):
    problem += pulp.lpSum(y[i,j,k] for i in range(n_point) for j in range(1,n_point)if i != j and k != j and i != k)<=1

for j in range(1,n_point):
    for k in range(1,n_point+1):
        if j !=k:
            problem += y[0,j,k]<=pulp.lpSum(x[h,k]for h in range(n_point)if h != k)

for j in range(1,n_point):
    problem += pulp.lpSum(x[i,j]for i in range(n_point) if i !=j) + pulp.lpSum(y[i,j,k]for i in range(n_point)for k in range(1,n_point+1)if k != j and i != j and i !=k)==1
#全ての頂点を回る

for j in range(1,n_point):
    problem +=pulp.lpSum(x[i,j]for i in range(n_point) if i != j)==pulp.lpSum(x[j,k]for k in range(1,n_point+1)if j != k)
#iから出発して、jに到着するなら、jから出発してに到着


for i in range(1,n_point):
    problem += a[i] >= t[i]- M*(1-pulp.lpSum(y[i,j,k]for j in range(1,n_point) for k in range(1,n_point+1) if i !=j and j != k and i !=k))


for i in range(1,n_point):
    problem += a[i] <= t[i]+M*(1-pulp.lpSum(y[i,j,k]for j in range(1,n_point) for k in range(1,n_point+1) if i !=j and j != k and i !=k))


for k in range(1,n_point+1):
    problem += a[k] >= t[k]-M*(1-pulp.lpSum(y[i,j,k]for i in range(n_point) for j in range(1,n_point) if i !=j and j != k and i !=k))

for k in range(1,n_point+1):
    problem += a[k] <= t[k]+M*(1-pulp.lpSum(y[i,j,k]for i in range(n_point) for j in range(1,n_point) if i !=j and j != k and i !=k))

for k in range(1,n_point):
    for h in range(n_point):
        if k != h:
            problem += t[k] >= t[h]+distances[h][k]/b+S_L*(pulp.lpSum(y[k,l,m]for l in range(1,n_point) for m in range(1,n_point+1) if l !=k and k != m and l != m))+S_R*(pulp.lpSum(y[i,j,k]for i in range(n_point) for j in range(1,n_point) if i !=j and j !=k and k != i))-M*(1-x[h,k])
for h in range(n_point):
    problem += t[n_point] >=t[h]+distances[h][n_point]/b+S_R*(pulp.lpSum(y[i,j,n_point]for i in range(n_point) for j in range(1,n_point) if i != j))-M*(1-x[h,n_point])

for j in range(1,n_point):
    for i in range(n_point):
        if i !=j:
            problem+= a[j] >= a[i]+distances[i][j]/c-M*(1-pulp.lpSum(y[i,j,k]for k in range(1,n_point+1)if j != k and i != k))

for j in range(1,n_point):
    for k in range(1,n_point+1):
        if j !=k:
            problem += a[k]>=a[j]+distances[j][k]/c+S_R-M*(1-pulp.lpSum(y[i,j,k]for i in range(n_point)if i != k and i !=j))

for k in range(1,n_point+1):
    for j in range(1,n_point):
        for i in range(n_point):
            if i != j and j != k and i != k:
                problem += a[k]-(a[j]-distances[i][j]/c)<=e+M*(1-y[i,j,k])


for i in range(1,n_point):
    for j in range(1,n_point):
        if i !=j:
            problem+=u[i]-u[j]>=1-(c+2)*p[i,j]
for i in range(1,n_point):
    for j in range(1,n_point):
        if i !=j:
            problem+=u[i]-u[j]<=-1+(c+2)*(1-p[i,j])
for i in range(1,n_point):
    for j in range(1,n_point):
        if i !=j: 
            problem+=p[i,j]+p[j,i]==1



for i in range(n_point):
    for k in range(1,n_point+1):
        for l in range(1,n_point):
            if i !=k and l !=i and l !=k:
                problem +=a[l]>=a[k]-M*(3-pulp.lpSum(y[i,j,k]for j in range(1,n_point)if l != k and j != k and j != i)-pulp.lpSum(y[l,m,n]for m in range(1,n_point)for n in range(1,n_point+1)if m !=i and m != l and m != k and n != i and n !=k)-p[i,l])

problem+=t[0]==0
problem+=a[0]==0
for j in range(1,n_point):
    problem+=p[0,j]==1



#最大飛行時間


for i in range(1,n_point):
    for K in range(1,n_point+1):        
        if  i!=k and (i != 0 and j != 0):
            problem += u[k] - u[i] >= 1 - (n_point+1) * (1 - pulp.lpSum(y[i,j,k]for j in range(1,n_point)))

# eliminate subtour
for i in range(1,n_point):
    for j in range(1,n_point+1):        
        if i != j and (i != 0 and j != 0):
            problem += u[i] - u[j] <= (n_point+1) * (1 -x[i, j])- 1

problem +=pulp.lpSum((distances[i][j]+distances[j][k])*y[i,j,k]for i in range(n_point) for j in range(1,n_point) for k in range(1,n_point+1) if i != j and j != k and i !=k )/c <= e

# solve problem
status = problem.solve()

print(pulp.LpStatus[status])
# output status, value of objective function
status, pulp.LpStatus[status], pulp.value(problem.objective)


# check TSP problem and optimized route

plt.figure(figsize=(5, 5))

# draw problem state
for i, row in df.iterrows():
    if i == 0:
        plt.scatter(row['x'], row['y'], c='r')
        plt.text(row['x'] + 1, row['y'] + 1, 'depot')
        
    else:
        plt.scatter(row['x'], row['y'], c='black')
        plt.text(row['x'] + 1, row['y'] + 1, f'{i}')
        
plt.xlim([-10, 50])
plt.ylim([-10, 50])
plt.title('points: id')

# draw optimal route
routes = [(i, j) for i in range(n_point) for j in range(1,n_point+1) if pulp.value(x[i, j]) == 1]
arrowprops = dict(arrowstyle='->', connectionstyle='arc3', edgecolor='blue')
for i, j in routes:
    plt.annotate('', xy=[df.iloc[j]['x'], df.iloc[j]['y']], xytext=[df.iloc[i]['x'], df.iloc[i]['y']], arrowprops=arrowprops)

routes1 =[(i, j,k) for i in range(n_point) for j in range(1,n_point) for k in range(1,n_point+1) if pulp.value(y[i, j,k]) == 1]

arrowprops = dict(arrowstyle='->', connectionstyle='arc3', edgecolor='red')
for i, j,k in routes1:
    plt.annotate('', xy=[df.iloc[j]['x'], df.iloc[j]['y']], xytext=[df.iloc[i]['x'], df.iloc[i]['y']] ,arrowprops=arrowprops)    
    plt.annotate('', xy=[df.iloc[k]['x'], df.iloc[k]['y']], xytext=[df.iloc[j]['x'], df.iloc[j]['y']] ,arrowprops=arrowprops)    
plt.show()
