#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import argparse

import numpy as np
import pandas as pd


def parse_args ():

  description = 'Pyron detector test example'

  parser = argparse.ArgumentParser(description = description)
  parser.add_argument('--data',
                      dest='data_cfg',
                      required=True,
                      type=str,
                      action='store',
                      help='Data cfg filename'
                      )
  parser.add_argument('--cfg',
                      dest='netcfg',
                      required=False,
                      type=str,
                      action='store',
                      help='Configuration network filename/path',
                      default=''
                      )
  parser.add_argument('--weights',
                      dest='weights',
                      required=False,
                      type=str,
                      action='store',
                      help='Weights filename/path',
                      default='')
  parser.add_argument('--mask',
                      dest='maskfile',
                      required=False,
                      type=str,
                      action='store',
                      help='Maskfile of weights filename/path',
                      default=''
                      )
  parser.add_argument('--names',
                      dest='namesfile',
                      required=False,
                      type=str,
                      action='store',
                      help='Labels (names) filename/path',
                      default=''
                      )
  parser.add_argument('--input',
                      dest='input',
                      required=False,
                      type=str,
                      action='store',
                      help='Input filename/path',
                      default=''
                      )
  parser.add_argument('--output',
                      dest='outfile',
                      required=False,
                      type=str,
                      action='store',
                      help='Output filename/path',
                      default=''
                      )
  parser.add_argument('--thresh',
                      dest='thresh',
                      required=False,
                      type=float,
                      action='store',
                      help='Probability threshold prediction',
                      default=.5
                      )
  parser.add_argument('--hierthresh',
                      dest='hier',
                      required=False,
                      type=float,
                      action='store',
                      help='Hierarchical threshold',
                      default=.5
                      )
  parser.add_argument('--classes',
                      dest='classes',
                      required=False,
                      type=int,
                      action='store',
                      help='Number of classes to read',
                      default=-1
                      )
  parser.add_argument('--nms',
                      dest='nms',
                      required=False,
                      type=float,
                      action='store',
                      help='Threshold of IOU value',
                      default=.45
                      )
  parser.add_argument('--save',
                      dest='save',
                      required=False,
                      type=bool,
                      action='store',
                      help='Save the image with detection boxes',
                      default=False
                      )
  parser.add_argument('--nth',
                      dest='nth',
                      required=False,
                      type=int,
                      action='store',
                      help='Number of thread to use',
                      default=NTH
                      )

  args = parser.parse_args()

  return args

def main():

  args = parser.parse_args()


if __name__ == '__main__':
  main()
