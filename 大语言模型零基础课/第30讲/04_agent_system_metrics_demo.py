# 第30讲实验：Agent System Metrics Demo
# 用一个小型合成结果表，演示为什么 Agent 研究不能只报告最终准确率。

from itertools import combinations
import math

# 12 tasks, 3 agents. 1=correct, 0=wrong
A = [1,1,1,0,1,0,1,1,0,1,0,1]
B = [1,1,0,1,1,0,1,0,1,1,0,1]
C = [1,0,1,1,1,0,0,1,1,1,0,1]

# toy resource usage for each agent across all tasks
TOKENS = {'A': 7200, 'B': 7600, 'C': 7100}
CALLS  = {'A': 24,   'B': 26,   'C': 23}

agents = {'A':A, 'B':B, 'C':C}

def accuracy(xs):
    return sum(xs)/len(xs)

def majority_vote(rows):
    out=[]
    for vals in zip(*rows):
        out.append(1 if sum(vals) > len(vals)/2 else 0)
    return out

def pearson_binary_error(x,y):
    ex=[1-v for v in x]
    ey=[1-v for v in y]
    mx=sum(ex)/len(ex); my=sum(ey)/len(ey)
    num=sum((a-mx)*(b-my) for a,b in zip(ex,ey))
    dx=math.sqrt(sum((a-mx)**2 for a in ex))
    dy=math.sqrt(sum((b-my)**2 for b in ey))
    return num/(dx*dy) if dx and dy else 0.0

print('Agent System Metrics Demo')
print('='*72)
for name,vals in agents.items():
    print(f'{name}: accuracy={accuracy(vals):.3f} tokens={TOKENS[name]} calls={CALLS[name]}')

maj=majority_vote([A,B,C])
mean_individual=sum(accuracy(v) for v in agents.values())/len(agents)
maj_acc=accuracy(maj)
print()
print(f'Mean individual accuracy: {mean_individual:.3f}')
print(f'Majority-vote accuracy:    {maj_acc:.3f}')
print(f'Collective gain:           {maj_acc-mean_individual:+.3f}')

print('\nPairwise error correlations:')
cors=[]
for i,j in combinations(agents,2):
    c=pearson_binary_error(agents[i],agents[j])
    cors.append(c)
    print(f'  {i}-{j}: {c:.3f}')
print(f'Average error correlation: {sum(cors)/len(cors):.3f}')

print('\nTotal system resources:')
print('  tokens =',sum(TOKENS.values()))
print('  calls  =',sum(CALLS.values()))

print('\nLesson:')
print('- Outcome metric: majority-vote accuracy')
print('- Mechanism metric: error correlation')
print('- Cost metric: tokens / calls')
print('- A convincing experiment usually needs all three.')
