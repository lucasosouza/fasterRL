
import shutil
import os

for folder in ['logs', 'results', 'runs', 'weights']:
    shutil.rmtree('/Users/lucasosouza/Documents/fasterrldata/{}'.format(folder))
    os.mkdir('/Users/lucasosouza/Documents/fasterrldata/{}'.format(folder))
