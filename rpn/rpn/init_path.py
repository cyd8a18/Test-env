import sys
import os
path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(path, '../'))

print(path)
print(os.path.join(path, '../'))