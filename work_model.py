
from data import project, run, config, device

def train(dataloader, steps, model, optim, fun_loss, flag=True):
    model.train()
  #  print(model)
  #  i = 0
    for batch, (x, y) in enumerate(dataloader):
        optim.zero_grad()
        x = x.to(device)
        y = y.to(device)
        ans = model(x)
        loss = fun_loss(ans, y)
        print(x)
        print(y)
        print(ans)
        print(loss)
        if batch == 10:
            break
        loss.backward()
        optim.step()
        step =  steps + (1 + batch) / len(dataloader)
        if flag and batch % 1 == 0:
            run['losses'].log(loss.item(), step=step)



def test(dataloader, step, model, fun_loss, flag=True):
    model.eval()
    accur = 0
    sum_loss = 0
    cnt = 0 
    with torch.no_grad():
      for x, y in dataloader:
          x = x.to(device)
          y = y.to(device)
          ans = model(x)
          loss = fun_loss(ans, y)
          sum_loss += loss.item()
          cnt += x.shape[0]
          accur += (ans.argmax(dim=1) == y).type(torch.float).sum().item()

    accur /= cnt
    sum_loss /= len(dataloader)
    if flag:
        run['losses_test'].log(sum_loss, step=step)
        run['accuracy'].log(accur, step=step)
    return (accur, sum_loss)