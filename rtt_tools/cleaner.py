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


logger = logging.getLogger(__name__)
coloredlogs.CHROOT_FILES = []
coloredlogs.install(level=logging.DEBUG, use_chroot=False)
EMPTY_SHA = b'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
EMPTY_SHA_STR = EMPTY_SHA.decode('utf8')


def try_execute(fnc, attempts=40, msg=""):
    for att in range(attempts):
        try:
            return fnc()

        except Exception as e:
            logger.error("Exception in executing function, %s, att=%s, msg=%s" % (e, att, msg))
            if att - 1 == attempts:
                raise
    raise ValueError("Should not happen, failstop")


class Cleaner:
    def __init__(self):
        self.args = None
        self.conn = None
        self.exp_id_low = None

    def proc_args(self, args=None):
        parser = argparse.ArgumentParser(description='RTT result cleaner')
        parser.add_argument('--small', dest='small', action='store_const', const=True, default=False,
                            help='Small result set (few experiments)')
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
        with self.conn.cursor() as c:
            logger.info("Processing experiments")

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

    def fix_mpcexps(self):
        broken_exps = ["testmpc02-lowmc-s80a-bin-raw-r5-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r10-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r90-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r11-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r20-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r3-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r9-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r8-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r4-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r11-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r5-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r11-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r190-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r5-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r210-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r70-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r20-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r130-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r6-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r8-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r10-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r5-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r4-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r6-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r200-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r10-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r7-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r11-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r3-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r50-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r4-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r90-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r110-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r10-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r8-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r30-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r3-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r90-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r5-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r220-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r9-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r4-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r150-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r230-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r90-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r6-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r7-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r8-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r110-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r7-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r6-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r70-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r8-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r70-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r40-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r70-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r30-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r3-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r10-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r80-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r5-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r20-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r120-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r20-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r4-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r60-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r9-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r7-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r160-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r70-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r240-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r4-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r100-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r170-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r7-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r7-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r11-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r8-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r8-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r140-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r8-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r20-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r50-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r9-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r9-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r9-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r160-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r11-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r170-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r7-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r200-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r11-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r7-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r10-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r5-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r6-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r5-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r9-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r30-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r3-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r10-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r180-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r40-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r4-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r8-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r50-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r7-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r100-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r5-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r130-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r40-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r7-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r80-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r80-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r5-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r30-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r140-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r60-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r30-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r11-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r30-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r3-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r11-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r60-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r4-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r60-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r5-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r100-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r11-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r8-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r4-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r40-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r3-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r6-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r60-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r10-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r9-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r180-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r10-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r9-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r100-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r210-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r6-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r11-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r20-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r6-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r8-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r6-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r11-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r4-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r80-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r60-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r3-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r6-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r240-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r4-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r10-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r190-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r50-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r6-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r110-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r3-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r8-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r150-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r5-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r3-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r6-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r3-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r9-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s128d-bin-raw-r50-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r9-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r230-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r120-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r70-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r9-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r50-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r80-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128c-bin-raw-r40-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r10-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r7-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r40-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r7-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r3-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s128a-bin-raw-r4-inp-lhw03-b32-w4-spr--s100MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r80-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r110-inp-lhw03-b16-w5-spr--s100MB.json",
                       "testmpc02-lowmc-s80a-bin-raw-r10-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r220-inp-lhw03-b16-w4-spr--s10MB.json"]

        def rename_lhw(inp, to):
            res = re.sub(r'-inp-([\w]+?)([\d]+)-', '-inp-\\1!XXXX-', inp)
            return res.replace('!XXXX', '%02d' % to)

        tests_all = json.load(open('testmpc02-exps.json'))
        renames = []
        for broken in broken_exps:
            regex = broken.replace('.json', '')
            regex = re.sub(r'-inp-([\w]+?)([\d]+)-', '-inp-\\1[\\\\d]+-', regex)
            sel = []
            for ctest in tests_all:
                if re.match(regex, ctest):
                    m = re.match(r'.*-inp-([\w]+?)([\d]+)-.*', ctest)
                    sel.append((ctest, int(m.group(2))))

            seqs = [x[1] for x in sel]
            seqss = sorted(seqs)

            if seqss == [0, 2, 3]:
                pass
            elif seqss == [1, 2, 3]:
                pass
            else:
                print(f'Broken: {broken} all related: {[x[1] for x in sel]}, {sel}')
                continue

            for replace_ix in range(1, 4):
                csel = [x for x in sel if x[1] == replace_ix]
                if not csel:
                    continue
                renames.append((replace_ix, rename_lhw(csel[0][0], replace_ix - 1), csel[0][0]))

        renames.sort(key=lambda x: x[0])
        renames_map = {x[2]: (x[0], x[1]) for x in renames}

        with self.conn.cursor() as c:
            logger.info("Processing experiments")

            c.execute("""SELECT id, name, status, created, run_started, run_finished, data_file_sha256, data_file_size, 
                                 data_file_id FROM experiments 
                                 WHERE id >= %s
                              """ % (self.exp_id_low,))

            eid_data = {}
            for result in c.fetchall():
                eid, name, status, sha, fsize = result[0], result[1], result[2], result[6], result[7]
                if name in renames_map:
                    eid_data[name] = eid

            for ren in renames:
                if ren[2] not in eid_data:
                    continue
                eid = eid_data[ren[2]]

                sql_exps = 'UPDATE experiments SET name=%s WHERE id=%s'
                print(f'Updating {eid} to {ren[1]}')
                # try_execute(lambda: c.execute(sql_exps, (ren[1], eid,)),
                #             msg="Update experiment with ID %s" % eid)

            self.conn.commit()
            logger.info('Experiment %s solved' % (eid,))

    def fix_colon(self):
        with self.conn.cursor() as c:
            logger.info("Processing experiments")

            c.execute("""SELECT id, name, status, created, run_started, run_finished, data_file_sha256, data_file_size, 
                                 data_file_id FROM experiments 
                                 WHERE id >= %s
                              """ % (self.exp_id_low,))

            renames = []
            for result in c.fetchall():
                eid, name, status, sha, fsize = result[0], result[1], result[2], result[6], result[7]
                if 'stream:cipher' in name:
                    renames.append((eid, name))

            for eid, name in renames:
                nname = name.replace('stream:cipher', 'stream_cipher')

                sql_exps = 'UPDATE experiments SET name=%s WHERE id=%s'
                print(f'Updating {eid} to {nname}')
                try_execute(lambda: c.execute(sql_exps, (nname, eid,)),
                            msg="Update experiment with ID %s" % eid)

            self.conn.commit()
            logger.info('Experiment %s solved' % (eid,))

    def fix_lowmc(self):
        with self.conn.cursor() as c:
            logger.info("Processing experiments")

            c.execute("""SELECT e.id, name, dp.`provider_config`
                            FROM experiments e
                            JOIN rtt_data_providers dp ON e.data_file_id = dp.id
                            WHERE e.id >= %s and name LIKE '%%lowmc-s80a%%'
                              """ % (self.exp_id_low,))

            renames = []
            for result in c.fetchall():
                eid, name, config, = result[0], result[1], result[2]
                config_js = json.loads(config)
                block_size = config_js['input_files']['CONFIG1.JSON']['data']['stream']['block_size_bits']
                if block_size == 128:
                    renames.append((eid, name, name.replace('lowmc-s80a', 'lowmc-s80b')))

            for eid, oname, nname in renames:
                sql_exps = 'UPDATE experiments SET name=%s WHERE id=%s'
                print(f'Updating {eid} to {nname}')
                try_execute(lambda: c.execute(sql_exps, (nname, eid,)),
                            msg="Update experiment with ID %s" % eid)

            self.conn.commit()
            logger.info('Experiment %s solved' % (eid,))


def main():
    l = Cleaner()
    l.main()


if __name__ == "__main__":
    main()
