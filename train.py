from __future__ import print_function
from options import Options
from data import load_data
from model import Ganomaly

opt = Options().parse()
dataloader = load_data(opt)
model = Ganomaly(opt, dataloader)

model.train()
# model.select_threshold()
# model.validate(threshold_value=0.06)
