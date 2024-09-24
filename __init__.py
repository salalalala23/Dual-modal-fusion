import torch
import argparse

DATA_DICT = {  # 数据集对应的数据图像尺寸大小和对应的颜色
    'image4': {
        "size": [800, 830, 4],
        "color": [[0, 0, 0], [255, 192, 203], [255, 165, 0], [0, 255, 255],
                  [255, 0, 0], [160, 80, 43], [123, 255, 0], [0, 0, 255]]},
    'image5': {
        "size": [4541, 4548, 4],
        "color": [[0, 0, 0], [0, 255, 255], [0, 0, 255], [0, 255, 127], [237, 145, 33],
                  [189, 252, 201], [255, 0, 0], [139, 58, 58], [160, 32, 240],
                  [221, 160, 221], [240, 230, 140], [255, 0, 255], [255, 255, 0]]},
    'image6': {
        "size": [2001, 2101, 4],
        "color": [[0, 0, 0], [0, 255, 255], [0, 0, 255], [237, 145, 33],
                  [0, 255, 0], [160, 32, 240], [221, 160, 221], [240, 230, 140],
                  [255, 0, 0], [255, 255, 0], [0, 255, 127], [255, 0, 255]]},
    'image7': {
        "size": [2000, 2500, 4],
        "color": [[0, 0, 0], [0, 255, 255], [0, 0, 255], [237, 145, 33],
                  [0, 255, 0], [240, 230, 140], [255, 0, 0], [160, 32, 240],
                  [255, 255, 0], [221, 160, 221], [0, 255, 127], [255, 0, 255]]},
    'image8': {
        "size": [3408, 4000, 4],
        "color": [[0, 0, 0], [0, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0],
                  [221, 160, 221], [240, 230, 140], [237, 145, 33], [0, 128, 0]]},
    'image9': {
        "size": [6905, 7300, 4],
        "color": [[0, 0, 0], [0, 255, 255], [0, 0, 255], [237, 145, 33], [189, 252, 201],
                  [230, 0, 0], [0, 255, 0], [160, 32, 240], [221, 160, 221],
                  [240, 230, 140], [153, 153, 0]]},
    'image10': {
        "size": [6905, 7300, 4],
        "color": [[0, 0, 0], [0, 255, 255], [0, 0, 255], [237, 145, 33], [189, 252, 201],
                  [230, 0, 0], [0, 255, 0], [160, 32, 240], [221, 160, 221],
                  [240, 230, 140], [153, 153, 0]]},
}

p = argparse.ArgumentParser(description="This is Dual-modal-fusion")
p.add_argument("--bs", type=int, default=64, help='batch size')
p.add_argument("--bs_t", type=int, default=300, help='batch size for test')
p.add_argument("--tr", type=float, default=0.05, help='train rate')
p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--ps", type=int, default=16, help='patchsize')
p.add_argument("--ep", type=int, default=50, help='epoch')
p.add_argument("--time", type=int, default=5, help="number of experimental replicates")
p.add_argument("--lr", type=int, default=1e-3, help='learning rate')
p.add_argument("--parameter", type=str, default='tr'+str(p.parse_args().tr)+'_ep'+
                                                str(p.parse_args().ep)+'_bs'+str(p.parse_args().bs)+'/')

p.add_argument("--data_city", type=str, default='image6')
p.add_argument("--address", type=str, default=r'H:/Code/')
p.add_argument("--remote_data", type=str, default=p.parse_args().address + 'Remote data/')
p.add_argument("--cur_file", type=str, default=str(__file__.rsplit("\\", 2)[-2]))
p.add_argument("--expo_result", type=str, default=p.parse_args().address + p.parse_args().cur_file + '/')
p.add_argument("--result", type=str, default=p.parse_args().address+'/Expo_result/' +
                                             p.parse_args().cur_file + "/" + p.parse_args().data_city + '/')

p.add_argument("--Categories", type=int, default=len(DATA_DICT[p.parse_args().data_city]["color"]))
p.add_argument("--size", type=list, default=DATA_DICT[p.parse_args().data_city]['size'])
p.add_argument("--colormap", type=list, default=DATA_DICT[p.parse_args().data_city]['color'])

p.add_argument('--name', type=str, help='your name')
args = p.parse_args()