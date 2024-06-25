import glob, os
from math import *
from tqdm import tqdm
import shutil

input_folders = [
    'flowers/daisy',
    'flowers/dandelion',
    'flowers/roses',
    'flowers/sunflowers',
    'flowers/tulips',
]

BASE_DIR_ABSOLUTE = "D:\\MyProjects\\Data Sciense project\\My-first-AI"
OUT_DIR = "./flowers/"

OUT_TRAIN = OUT_DIR + "train/"
OUT_VAL = OUT_DIR + "test/"

coeff = [80, 20]
exceptions = ['classes']

if int(coeff[0]) + int(coeff[1]) > 100:
    print("Coeff can't exceed 100%")
    exit(1)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

print(f"Preparing images data by {coeff[0]}/{coeff[1]} rule")
print(f"Source folders: {len(input_folders)}")
print(f"Gathering data........")

source = {}
for sf in input_folders:
    source.setdefault(sf, [])

    os.chdir(BASE_DIR_ABSOLUTE)
    os.chdir(sf)

    for filename in glob.glob("*.jpg"):
        source[sf].append(filename)

train = {}
val = {}
for sk,sv in source.items():
    chunks = 10
    train_chunks = floor(chunks * coeff[0] / 100)
    val_chunks = chunks - train_chunks

    train.setdefault(sk, [])
    val.setdefault(sk, [])

    for item in chunker(sv, chunks):
        train[sk].extend(item[0:train_chunks])
        val[sk].extend(item[train_chunks:])

train_sum = 0
val_sum = 0

for sk, sv in train.items():
    train_sum += len(sv)

for sk, sv in val.items():
    val_sum += len(sv)
print(f"\nOverall TRAIN images count: {train_sum}")
print(f"Overall test images count: {val_sum}")

os.chdir(BASE_DIR_ABSOLUTE)
print(f"\nCopying TRAIN source items to prepare folders........")

for sk, sv in tqdm(train.items()):
    for item in tqdm(sv):
        imgfile_source = os.path.join(BASE_DIR_ABSOLUTE, sk, item)
        imgfile_dest = os.path.join(OUT_TRAIN, sk.split("/")[-2], '')

        os.makedirs(imgfile_dest, exist_ok=True)
        shutil.copyfile(imgfile_source, os.path.join(imgfile_dest, item))

os.chdir(BASE_DIR_ABSOLUTE)
print(f"\nCopying VAL source items to prepare folders........")

for sk, sv in tqdm(val.items()):
    for item in tqdm(sv):
        imgfile_source = os.path.join(BASE_DIR_ABSOLUTE, sk, item)
        imgfile_dest = os.path.join(OUT_VAL, sk.split("/")[-2], '')

        os.makedirs(imgfile_dest, exist_ok=True)
        shutil.copyfile(imgfile_source, os.path.join(imgfile_dest, item))

print(f"\nDone")