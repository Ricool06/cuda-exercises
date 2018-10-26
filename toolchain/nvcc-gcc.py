#!/usr/bin/env python3
from argparse import ArgumentParser
import subprocess
parser = ArgumentParser(description='no')
parser.add_argument('-c', nargs='+')
parser.add_argument('-o', nargs='+')
args, extra = parser.parse_known_args()

print(args.c)

srcs = ' '.join(args.c)
outs = ' '.join(args.o)

subprocess.call([
  '/usr/local/cuda/bin/nvcc',
  '--compile',
  '-c', srcs,
  '-o', outs,
  '--compiler-options', ' '.join(extra)])
