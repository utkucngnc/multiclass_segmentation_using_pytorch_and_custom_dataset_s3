from dataset_sock import dataset_sock as ds
from torch.utils.data import DataLoader as dl
from tqdm import tqdm

aa = ds(4)

bb = dl(aa, 1, False)

for i, elem in enumerate(bb):
    print(f'i:{i} and elem:{elem}')

cc =tqdm(bb)