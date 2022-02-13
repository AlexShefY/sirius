import plotly.express as px
import numpy as np

def im_show(pic):
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


if __name__ == '__main__':
    
    from data import build_dataloader
    train_dataloader, val_dataloader, test_dataloader = build_dataloader()
    import random

    i = random.randint(0, 1000)
    for j in range(10):
        im_show(val_dataloader.dataset[j + i][0])
