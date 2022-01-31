import torch
from data import project, run, config, device
from torchvision.transforms import Grayscale, autoaugment, AutoAugment
import numpy as np

augmenter = AutoAugment(autoaugment.AutoAugmentPolicy.SVHN)
def prepare_data(x):
  return augmenter((x* 255).to(torch.uint8)).to(torch.float) / 255

def gray(x):
  return Grayscale()((x * 255).to(torch.uint8)).to(torch.float) / 255

def cat_one(image, size=4, n_squares=1):
    h, w, channels = image.shape
    new_image = image
    for _ in range(n_squares):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - size // 2, 0, h)
        y2 = np.clip(y + size // 2, 0, h)
        x1 = np.clip(x - size // 2, 0, w)
        x2 = np.clip(x + size // 2, 0, w)
        v = torch.sum(new_image[:, x1:x2, y1:y2], (2, 1)) / (size ** 2)
        for k in range(3):
          new_image[k, x1:x2, y1:y2] = v[k]
    return new_image

def cat_out(image, size=4, n_squares=0):
  new_image = image
  for i in range(image.shape[0]):
    new_image[i] = cat_one(image[i], size,n_squares)
  return new_image

from torchvision.transforms import ColorJitter, RandomPerspective

import random
brightness = 0.04835025953743052
contrast = 0.07352095579562219
hue = 0.01536353254455466
distortion_scale = 0.009800950085236237
p = 0.09129580741470969

run['brightness'] = brightness
run['contrast'] = contrast
run['hue'] = hue
run['distortion_scale'] = distortion_scale
run['p'] = p

print(brightness, contrast, hue)

jitter = ColorJitter(brightness=brightness
, contrast=contrast
, saturation=0, hue=hue
)
perspective = RandomPerspective(distortion_scale, p)
def train(dataloader, steps, model, optim, fun_loss, flag=True):
    model.train()
    sm = 0.0
    cn = 0.0
    for batch, (x, y) in enumerate(dataloader):
        optim.zero_grad()
        x = cat_out(jitter(perspective(x.to(device))))
        y = y.to(device)
        ans = model(x)
        sm += (ans.argmax(dim=1) == y).type(torch.float).sum().item()
        cn += ans.shape[0]
        loss = fun_loss(ans, y)
        loss.backward()
        optim.step()
        step =  steps + (1 + batch) / len(dataloader)
        if flag and batch % 10 == 0:
            run['losses'].log(loss.item(), step=step)
    if flag:
      run['train_accuracy'].log(sm / cn, step=steps + 1)


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