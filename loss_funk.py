# creating a loss function
import torch

# x1 og x2 er array
def loss_funk (x1, x2):
    score = 0
    for i,j in zip(x1,x2):
        if i == j:
            score +=1
        else:
            score -=1
    return score



x1 = torch.rand(1,10)
x2 = torch.rand(1,10)
print("",x1,"\n",x2)

print(loss_funk(x1,x2))
