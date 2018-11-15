
import shutil
import os

for folder in ['logs', 'results', 'runs', 'weights']:
    shutil.rmtree('/Users/lucasosouza/Documents/fasterRLdata/{}'.format(folder))
    os.mkdir('/Users/lucasosouza/Documents/fasterRLdata/{}'.format(folder))
