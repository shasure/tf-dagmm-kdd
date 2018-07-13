#!/usr/bin/env python
# -*- coding: utf-8 -*-
from model.config import hyperparams
from model.dagmm import DAGMM

if __name__ == '__main__':
    # train and valid
    dagmm = DAGMM(hyperparams)
    dagmm.train(valid=True)
