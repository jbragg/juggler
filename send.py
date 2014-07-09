from __future__ import division
import requests
import os
import csv
import json
import argparse

SERVER_URL = 'http://some.server.url'
CHUNK_SIZE = 500

    
def send_tables(expname):
    # Load tables
    expdir = os.path.join('res', expname)
    tablesdir = os.path.join(expdir, 'tables')
    def tablepath(s):
        return os.path.join(tablesdir, expname+'-'+s+'.csv')
    
    run_hash = dict()
    with open(tablepath('runs'), 'r') as f:
        reader = csv.DictReader(f)
        sent_rows = 0
        rows = []
        print 'sending runs...'
        for i,row in enumerate(reader):
            if i % CHUNK_SIZE == 0 and i > 0:
                r = requests.post(SERVER_URL + '/load_runs', 
                                  data={'rows': json.dumps(rows)})
                sent_rows += len(rows)
                run_hash.update(r.json())
                print 'sent {} runs'.format(sent_rows)
                rows = []

            rows.append(row)

        if rows:
            r = requests.post(SERVER_URL + '/load_runs', 
                              data={'rows': json.dumps(rows)})
            sent_rows += len(rows)
            run_hash.update(r.json())
            print 'sent {} runs'.format(sent_rows)

     
    with open(tablepath('history'), 'r') as f:
        reader = csv.DictReader(f)
        sent_rows = 0
        rows = []
        print 'loading histories...'
        for i,row in enumerate(reader):
            if i % CHUNK_SIZE == 0 and i > 0:
                requests.post(SERVER_URL + '/load_histories',
                              data={'rows': json.dumps(rows)})
                sent_rows += len(rows)
                print 'sent {} histories'.format(sent_rows)
                rows = []

            row['run_id'] = run_hash[row['run_id']]
            rows.append(row)

        if rows:
            r = requests.post(SERVER_URL + '/load_histories',
            data={'rows': json.dumps(rows)})
            sent_rows += len(rows)
            print 'sent {} histories'.format(sent_rows)



 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('expname', type=str,
                        help='experiment name')
    #parser.add_argument('-p', '--policies', nargs='+', type=int,
    #                    help='list of policies')
    #parser.add_argument('-i', '--maxiter', default=-1, type=int,
    #                    help='maximum iterations/policy for detailed plots')
    args = parser.parse_args()
    send_tables(args.expname)
    #make_plots(args.expname, args.policies, args.maxiter)
