from data import build_dataloader

import plotly.express as px
import numpy as np

train_dataloader, val_dataloader, test_dataloader = build_dataloader()

def im_show(pic):
    print(pic.shape)
    to_plot = 32 * [[[]]]
    for a in range(32):
        to_plot[a] = 32 * [[]]
        for b in range(32):
            to_plot[a][b] = 3 * [0]
            to_plot[a][b][0] = pic[0][a][b].item()
            to_plot[a][b][1] = pic[1][a][b].item()
            to_plot[a][b][2] = pic[2][a][b].item()
    fig = px.imshow(to_plot)
    fig.show()


i = 0
for j in range(10):
    im_show(val_dataloader.dataset[j + i][0])
