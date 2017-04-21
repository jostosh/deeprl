from tkinter.filedialog import askopenfilename
import pickle as pkl
import sys
import pprint

with open(sys.argv[1], 'rb') as f:
    hp = pkl.load(f)

pprint.pprint(hp)
