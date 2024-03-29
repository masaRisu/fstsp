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
    'x':np.random.randint(0,40,n_point),
    'y':np.random.randint(0,40,n_point),
})
df.iloc[0]['x']=20
df.iloc[0]['y']=0

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



#定数の定義
a=50
m=1
e=1
S_R=1/60
problem = pulp.LpProblem('KUPC2018_C', pulp.LpMinimize)

# 変数とその定義域の定義
x=pulp.LpVariable.dicts('x',((i,j) for i in range(n_point) for j in range(n_point)),lowBound=0,upBound=1,cat='Binary')
y=pulp.LpVariable.dicts('y',((i,j) for i in range(n_point) for j in range(n_point)),lowBound=0,upBound=1,cat='Binary')
# we need to keep track of the order in the tour to eliminate the possibility of subtours
u = pulp.LpVariable.dicts('u', (i for i in range(n_point)), lowBound=1, upBound=n_point, cat='Integer')

# set objective function
problem += pulp.lpSum((distances[i][j])*(x[i,j])for i in range(n_point)for j in range(n_point))/40+S_R


problem += (pulp.lpSum(x[0,j]for j in range(1,n_point)))== 1


problem += (pulp.lpSum(x[j,0]for j in range(1,n_point))) == 1


#全ての頂点を回る


#トラックはdépôtを出発してデポにかえってくる


for j in range(1,n_point):
    problem += pulp.lpSum(x[i,j]for i in range(n_point) if i !=j) + pulp.lpSum(y[i,j]for i in range(n_point)if i != j)==1
#全ての頂点を回る

for j in range(n_point):
    problem +=pulp.lpSum(x[i,j]for i in range(n_point) if i != j)==pulp.lpSum(x[j,k]for k in range(n_point)if j != k)
#iから出発して、jに到着するなら、jから出発してに到着

#ドローンの配達先は全てドローンで行う

for j in range(1,n_point):
    problem += pulp.lpSum(y[i,j]for i in range(n_point) if i !=j) + pulp.lpSum(y[j,k]for k in range(1,n_point)if k != j)<=1

problem += pulp.lpSum(y[j,0]for j in range(1,n_point)) ==m

for j in range(1,n_point):
    problem +=pulp.lpSum(y[i,j] for i in range(n_point) if i != j)==pulp.lpSum(y[j,0]) 

#決定変数に関する制約

for i in range(n_point):
    for j in range(n_point):
        if i !=j:
            problem += e >= (distances[i][j]/a+distances[j][0]/a)*y[i,j]

for i in range(n_point):
    for j in range(n_point):        
        if i != j and (i != 0 and j != 0):
            problem += u[i] - u[j] <= (n_point+1) * (1 -x[i, j])- 1
# solve problem
status = problem.solve()

print(pulp.LpStatus[status])
# output status, value of objective function
status, pulp.LpStatus[status], pulp.value(problem.objective)


problem = pulp.LpProblem('tsp_mip', pulp.LpMinimize)

# set valiables
X = pulp.LpVariable.dicts('X', ((i, j) for i in range(n_point) for j in range(n_point)), lowBound=0, upBound=1, cat='Binary')
# we need to keep track of the order in the tour to eliminate the possibility of subtours
z = pulp.LpVariable.dicts('u', (i for i in range(n_point)), lowBound=1, upBound=n_point, cat='Integer')

# set objective function
problem += pulp.lpSum(distances[i][j] * X[i, j] for i in range(n_point) for j in range(n_point))/40

# set constrains
for i in range(n_point):
    problem += X[i, i] == 0

for i in range(n_point):
    problem += pulp.lpSum(X[i, j] for j in range(n_point)) == 1
    problem += pulp.lpSum(X[j, i] for j in range(n_point)) == 1

# eliminate subtour
for i in range(n_point):
    for j in range(n_point):
        if i != j and (i != 0 and j != 0):
            problem += z[i] - z[j] <= n_point * (1 - X[i, j]) - 1

# solve problem
status = problem.solve()

# output status, value of objective function
status, pulp.LpStatus[status], pulp.value(problem.objective)


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
routes = [(i, j) for i in range(n_point) for j in range(n_point) if pulp.value(x[i, j]) == 1]
arrowprops = dict(arrowstyle='->', connectionstyle='arc3', edgecolor='blue')
for i, j in routes:
    plt.annotate('', xy=[df.iloc[j]['x'], df.iloc[j]['y']], xytext=[df.iloc[i]['x'], df.iloc[i]['y']], arrowprops=arrowprops)
routes1 = [(i, j) for i in range(n_point) for j in range(n_point) if pulp.value(y[i, j]) == 1]

arrowprops = dict(arrowstyle='->', connectionstyle='arc3', edgecolor='red')
for i, j in routes1:
    plt.annotate('', xy=[df.iloc[j]['x'], df.iloc[j]['y']], xytext=[df.iloc[i]['x'], df.iloc[i]['y']], arrowprops=arrowprops)    
plt.show()
