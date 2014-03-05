#!/bin/env python

"""Convert jocr relational plot csv to excel-style csv"""

import sys
import csv
from collections import defaultdict

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'usage: {} in.csv out.csv'.format(sys.argv[0])
        sys.exit()

    with open(sys.argv[1],'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        d = defaultdict(list) 
        for row in reader:
            p = row['policy']
            d[p].append(row)

    with open(sys.argv[2],'w') as f:
        names = []
        for p in d:
            names += [n + ' (' + p + ')' for n in fieldnames[1:]]

        writer = csv.DictWriter(f, names)
        writer.writeheader()
        for i in xrange(len(d.values()[0])):
            row = dict()
            for p in d:
                row.update(dict((n + ' (' + p + ')', d[p][i][n]) for
                                n in fieldnames[1:]))
            writer.writerow(row)
