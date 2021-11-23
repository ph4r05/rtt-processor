#! /usr/bin/python3
# @author: Dusan Klinec (ph4r05)
# pandas matplotlib numpy networkx graphviz scipy dill configparser coloredlogs mysqlclient requests sarge cryptography paramiko shellescape

from rtt_tools.common.rtt_db_conn import *
import configparser
import re
import math
import logging
import coloredlogs
import itertools
import collections
import json
import argparse
from typing import Optional, List, Dict, Tuple, Union, Any, Sequence, Iterable, Collection
import secrets
import base64
import hashlib

logger = logging.getLogger(__name__)
coloredlogs.CHROOT_FILES = []
coloredlogs.install(level=logging.DEBUG, use_chroot=False)


def try_execute(fnc, attempts=40, msg=""):
    for att in range(attempts):
        try:
            return fnc()

        except Exception as e:
            logger.error("Exception in executing function, %s, att=%s, msg=%s" % (e, att, msg))
            if att - 1 == attempts:
                raise
    raise ValueError("Should not happen, failstop")


class UserGen:
    def __init__(self):
        self.args = None
        self.conn = None
        self.exp_id_low = None

    def proc_args(self, args=None):
        parser = argparse.ArgumentParser(description='RTT user gen')
        parser.add_argument('--ufile', dest='ufile',
                            help='user file')
        self.args, unparsed = parser.parse_known_args()
        logger.debug("Unparsed: %s" % unparsed)

        # args override
        if not args:
            return

        for k in args:
            setattr(self.args, k, args[k])

    def connect(self):
        cfg = configparser.ConfigParser()
        cfg.read("config.ini")
        self.conn = create_mysql_db_conn(cfg)

    def load_data(self):
        with open(self.args.ufile) as fh:
            users = [y for y in [x.strip() for x in fh.readlines()] if y]

        with self.conn.cursor() as c:
            logger.info("Processing")

            c.execute("""SELECT id, name, status, created, run_started, run_finished, data_file_sha256, data_file_size, 
                         data_file_id FROM experiments 
                         WHERE id >= %s
                      """ % (self.exp_id_low, ))

            susp_data = []
            for result in c.fetchall():
                eid, name, status, sha, fsize = result[0], result[1], result[2], result[6], result[7]
                data_file_id = result[8]

                if status == 'pending' or status == 'finished':
                    continue
                if sha is None and fsize is None:
                    continue
                if (fsize is not None and fsize >= 1024) or (sha is not None and sha != EMPTY_SHA_STR):
                    continue

                if data_file_id is None:
                    logger.info('Suspicious ecp id %s, but empty file id %s' % (eid, result,))
                    continue

                logger.info('Suspicious experiment result: %s, fsize: %s, sha: %s'
                            % (eid, fsize, sha))

                # Delete batteries data
                c.execute('SELECT id FROM batteries WHERE experiment_id=%s', (eid,))
                bat_ids = [x[0] for x in c.fetchall()]
                logger.info('battery: %s' % bat_ids)

                for ix, bit in enumerate(bat_ids):
                    logger.info('Deleting battery records %s for expid %s' % (bit, eid))
                    try_execute(lambda: c.execute('DELETE FROM batteries WHERE id=%s', (bit,)),
                                msg="Delete batteries records for expid %s, bID: %s" % (eid, bit))

                # Update jobs
                logger.info('Update jobs')
                sql_jobs = 'UPDATE jobs SET status="pending", run_finished=NULL, run_heartbeat=NULL, retries=0 ' \
                           'WHERE experiment_id=%s'
                try_execute(lambda: c.execute(sql_jobs, (eid,)),
                            msg="Update jobs for expid %s" % eid)

                # Update experiment if finished
                logger.info('Update experiments')
                nstatus = 'pending'
                nhash = None if data_file_id is not None else sha
                nfsize = None if data_file_id is not None else fsize
                sql_exps = 'UPDATE experiments SET status=%s, run_started=NULL, run_finished=NULL, ' \
                           'data_file_sha256=%s, data_file_size=%s ' \
                           'WHERE id=%s'
                try_execute(lambda: c.execute(sql_exps, (nstatus, nhash, nfsize, eid,)),
                            msg="Update experiment with ID %s" % eid)

                self.conn.commit()
                logger.info('Experiment %s solved' % (eid,))

    def init(self, args=None):
        self.proc_args(args)
        self.connect()

    def load(self, args=None):
        self.init(args)

        tstart = time.time()
        self.load_data()
        logger.info('Time finished: %s' % (time.time() - tstart))

    def main(self, args=None):
        self.load(args)


def gen_password():
    passwd = secrets.token_urlsafe(18)
    salt = secrets.token_urlsafe(12)
    res = base64.b64encode(hashlib.pbkdf2_hmac('sha256', passwd.encode('ascii'), salt.encode('ascii'), 100000))
    return passwd, 'pbkdf2_sha256$100000$%s$%s' % (salt, res.decode('ascii'))


"""
import secrets
import base64
import hashlib
import json

db = [(u, *gen_password()) for u in users]
dbd = []

for ux in db:
    dbd.append(f'("{ux[2]}",NULL,0,"x{ux[0]}","x{ux[0]}","x{ux[0]}","{ux[0]}@mail.muni.cz", 0, 1, NOW())')
   
sql = "INSERT INTO auth_user(password, last_login, is_superuser, username, first_name, last_name, email, is_staff, is_active, date_joined) VALUES " + ",".join(dbd) 

db2=[[f'x{u[0]}',u[1]] for u in db]
print(json.dumps(db2, indent=2))

with(open('/tmp/rtt_accs.txt', 'w+')) as fh:
    json.dump(db2, fh, indent=2)
"""


def main():
    l = UserGen()
    l.main()


if __name__ == "__main__":
    main()

