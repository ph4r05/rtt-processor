#! /usr/bin/python3
# @author: Dusan Klinec (ph4r05)
# pandas matplotlib numpy networkx graphviz scipy dill configparser coloredlogs mysqlclient requests sarge cryptography paramiko shellescape
import hashlib
import os
import shutil

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
import random
from typing import Optional, List, Dict, Tuple, Union, Any, Sequence, Iterable, Collection

from rtt_tools.dump_data import chunks
from rtt_tools import generator_mpc as ggen
from rtt_tools.generator_mpc import get_input_key, get_input_size, comp_hw_weight, make_hw_config, HwConfig, \
    comb_cached, rank, HwSpaceTooSmall, generate_cfg_col, StreamOptions, ExpRec, write_submit_obj, generate_cfg_inp, \
    gen_lowmc_core, gen_script_config, MPC_SAGE_PARAMS
from rtt_tools.gen.max_rounds import FUNC_DB, FuncDb, FuncInfo
from rtt_tools.utils import natural_sort_key, merge_pvals

logger = logging.getLogger(__name__)
coloredlogs.CHROOT_FILES = []
coloredlogs.install(level=logging.DEBUG, use_chroot=False)
EMPTY_SHA = b'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
EMPTY_SHA_STR = EMPTY_SHA.decode('utf8')

"""
Useful queries:

# --------------------------------------------------------------------------------------------------------------------------------------------
# Check Booltest2 result, has to contain data SHA256 in the variant result (data file was correct when booltest2 was computed)

SELECT b.id AS bid, j.id AS jid, e.id AS eid, b.total_tests, b.passed_tests, b.name, e.name, j.run_started, data_file_sha256, vs.message
 FROM experiments e
 JOIN batteries b ON b.experiment_id = e.id
 JOIN jobs j ON b.job_id = j.id
 LEFT JOIN tests t ON t.battery_id = b.id
 LEFT JOIN variants v ON v.test_id = t.id
 LEFT JOIN variant_results vs ON vs.variant_id = v.id
 WHERE e.id > 284000 
  AND DATE(j.run_started) IN ('2022-01-15', '2022-01-16', '2022-01-17')
  AND b.name LIKE 'booltest_2%'
  AND t.name = 'halving 128-1-1'
  AND (e.data_file_sha256 is NULL OR INSTR(vs.message, e.data_file_sha256) = 0)
 ORDER BY e.id DESC
 LIMIT 1000
 
# --------------------------------------------------------------------------------------------------------------------------------------------
# Load results that have some batteries rejecting differently than others (disbalance)
 
SELECT b.id AS bid, j.id AS jid, e.id AS eid, b.total_tests, b.passed_tests, b.name, e.name, b2.id AS b2id, j.run_started, b2.name, b2.total_tests, b2.passed_tests 
 FROM experiments e
 JOIN batteries b ON b.experiment_id = e.id
 JOIN batteries b2 ON b2.experiment_id = e.id AND b.id != b2.id
 JOIN jobs j ON b.job_id = j.id
 WHERE e.id > 284000 
  AND b.name LIKE 'testu01%' AND b2.name LIKE 'testu01%'
  AND b.total_tests > 0 AND b2.total_tests > 0 
  AND (b.passed_tests / b.total_tests) < 0.5 and (b2.passed_tests / b2.total_tests) > 0.6
  #AND DATE(j.run_started) = '2022-01-16'
 ORDER BY e.id DESC LIMIT 5000
"""

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
        """Fix testmpc experiment seed run indexing, lhw03 was used.
        In those cases we need to reindex also lower runs"""
        broken_exps = ["testmpc02-lowmc-s80a-bin-raw-r5-inp-lhw03-b32-w3-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r10-inp-lhw03-b16-w4-spr--s10MB.json",
                       "testmpc02-lowmc-s128b-bin-raw-r90-inp-lhw03-b16-w4-spr--s10MB.json",
                       ]

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
        """Fix stream:cipher to stream_cipher in an experiment name"""
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

            logger.info(f'Renames to exec: {len(renames)}')
            random.shuffle(renames)
            for eid, name in renames:
                nname = name.replace('stream:cipher', 'stream_cipher')

                sql_exps = 'UPDATE experiments SET name=%s WHERE id=%s'
                print(f'Updating {eid} to {nname}')
                try_execute(lambda: c.execute(sql_exps, (nname, eid,)),
                            msg="Update experiment with ID %s" % eid)

            self.conn.commit()

    def fix_underscores(self):
        """Fix underscores to `:` in experiment name"""
        with self.conn.cursor() as c:
            logger.info("Processing experiments")

            c.execute("""SELECT id, name, status, created, run_started, run_finished, data_file_sha256, data_file_size, 
                                 data_file_id FROM experiments 
                                 WHERE id >= %s
                              """ % (self.exp_id_low,))

            renames = []
            for result in c.fetchall():
                eid, name, status, sha, fsize = result[0], result[1], result[2], result[6], result[7]
                if re.match(r'.*-t_[\w].*', name):
                    renames.append((eid, name))

            for eid, name in renames:
                nname = name
                nname = nname.replace('_', ':')
                nname = nname.replace('stream:cipher', 'stream_cipher')

                sql_exps = 'UPDATE experiments SET name=%s WHERE id=%s'
                print(f'Updating {eid} to {nname}')
                try_execute(lambda: c.execute(sql_exps, (nname, eid,)),
                            msg="Update experiment with ID %s" % eid)

            self.conn.commit()

    def fix_lowmc(self):
        """LowMC experiments had confused names (clash), just rename it"""
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

    def fix_lowmc_key(self):
        """LowMC experiments key naming convention refactoring"""
        with self.conn.cursor() as c:
            logger.info("Processing experiments")

            c.execute("""SELECT e.id, name, dp.`provider_config`
                            FROM experiments e
                            JOIN rtt_data_providers dp ON e.data_file_id = dp.id
                            WHERE e.id >= %s and name LIKE 'testmpc%%lowmc%%'
                              """ % (self.exp_id_low,))

            renames = []
            for result in c.fetchall():
                eid, name, config, = result[0], result[1], result[2]
                if '-inp-key.' not in name:
                    continue
                nname = re.sub(r'-inp-key\.(.+?)-', '-inp-\\1.key-', name)
                renames.append((eid, name, nname))

            for eid, oname, nname in renames:
                sql_exps = 'UPDATE experiments SET name=%s WHERE id=%s'
                print(f'Updating {eid} to {nname}')
                try_execute(lambda: c.execute(sql_exps, (nname, eid,)),
                            msg="Update experiment with ID %s" % eid)

            self.conn.commit()
            logger.info('Experiment %s solved' % (eid,))

    def fix_mpc_dups(self, from_id=None, dry_run=False):
        """Filter MPC duplicates"""
        with self.conn.cursor() as c:
            logger.info("Processing experiments")

            c.execute("""SELECT e.id, name, dp.`provider_config`
                            FROM experiments e
                            JOIN rtt_data_providers dp ON e.data_file_id = dp.id
                            WHERE e.id >= %s 
                            AND (name LIKE 'PH4-SM-%%' OR name LIKE 'testmpc%%')
                              """ % (from_id or self.exp_id_low,))

            dupes = []
            enames = {}
            for result in c.fetchall():
                eid, name, config, = result[0], result[1], result[2]
                if name in enames:
                    dupes.append([eid, name])
                    continue

                enames[name] = eid

            logger.info(f'Enames: {json.dumps(enames, indent=2)}')
            logger.info(f'Duplicates: {json.dumps(dupes, indent=2)}')

            for eids in chunks([str(x[0]) for x in dupes], 20):
                sql_exps = f'DELETE FROM jobs WHERE experiment_id IN ({",".join(eids)})'
                print(f'Deleting jobs for {eids} duplicate')
                if dry_run:
                    continue
                try_execute(lambda: c.execute(sql_exps, ()),
                            msg=f"Delete jobs for experiment with IDs {eids}")
            self.conn.commit()

            for eids in chunks([str(x[0]) for x in dupes], 20):
                sql_exps = f'UPDATE experiments SET status="finished" WHERE id IN ({",".join(eids)})'
                print(f'finishing {eids} duplicate')
                if dry_run:
                    continue
                try_execute(lambda: c.execute(sql_exps, (eid,)),
                            msg=f"update experiment with ID {eids}")
            self.conn.commit()

            for eids in chunks([str(x[0]) for x in dupes], 20):
                sql_exps = f'DELETE FROM experiments WHERE id IN ({",".join(eids)})'
                print(f'Deleting {eids} duplicate')
                if dry_run:
                    continue
                try_execute(lambda: c.execute(sql_exps, ()),
                            msg=f"Delete experiments with IDs {eids}")
                self.conn.commit()

            for name in enames:
                is_mpc = '-spr-' in name and '-raw-' in name
                if not is_mpc or not name.startswith('PH4-SM-'):
                    continue

                eid = enames[name]
                nname = re.sub(r'^PH4-SM-([\d]+)-', 'testmpc\\1-', name)
                sql_exps = 'UPDATE experiments SET name=%s WHERE id=%s'
                print(f'Updating {eid} to {nname}')
                if dry_run:
                    continue
                try_execute(lambda: c.execute(sql_exps, (nname, eid,)),
                            msg="Update experiment with ID %s" % eid)

            self.conn.commit()
            logger.info('Finished')

    def aux_mpc_booltest(self, from_id=None, tmpdir='/tmp/rspecs', skip_existing_since=None,
                        clean_before=False):
        """
        Add Booltest blocklens to match spreader sizes
        """
        if clean_before and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

        os.makedirs(tmpdir, exist_ok=True)

        agg_scripts = []
        computed_bools = collections.defaultdict(lambda: set())
        processed_data = collections.defaultdict()
        with self.conn.cursor() as c:
            logger.info("Processing experiments")
            existing_exps = self._load_existing_exps(c, skip_existing_since)

            c.execute("""SELECT e.id, name, dp.`provider_config`, dc.config_data, dc.config_name
                            FROM experiments e
                            JOIN rtt_data_providers dp ON e.data_file_id = dp.id
                            JOIN rtt_config_files dc ON e.config_file_id = dc.id
                            WHERE e.id >= %s and name LIKE 'testmpc%%' ORDER BY e.id
                              """ % (from_id or self.exp_id_low,))

            for result in c.fetchall():
                eid, name, config, rconfig, rcname = result[0], result[1], result[2], result[3], result[4]
                if 'rtt_data_gen.spreader' not in config:
                    continue

                cfg_hash = hashlib.sha256(config.encode('utf8')).hexdigest()
                m = re.match(r'.*\b(\d+)MB.*', rcname)
                if not m:
                    logger.error(f'Could not get exp size {eid}, {name}, {rcname}')
                    continue

                ssize = 1024 * 1024 * int(m.group(1))
                rtjs = json.loads(rconfig)
                cfg1 = rtjs['randomness-testing-toolkit'] if rtjs and 'randomness-testing-toolkit' in rtjs else None
                cfg2 = cfg1['booltest'] if cfg1 and 'booltest' in cfg1 else None
                cfg3 = cfg2['strategies'][0]['variations'][0]['bl'] if cfg2 and 'strategies' in cfg2 else None

                m = re.match(r'.*--ob=?(\d+)\b.*', config, flags=re.MULTILINE | re.DOTALL)
                if not m:
                    logger.warning(f'Could not parse spr: {eid}, {name}, {config}')
                    continue

                spr_obits = int(m.group(1))

                s_computed = set(cfg3) if cfg3 else set()
                computed_bools[cfg_hash] = computed_bools[cfg_hash].union(s_computed)
                processed_data[cfg_hash] = (ssize, spr_obits, eid, name, config, rtjs)

            for cfg_hash in processed_data.keys():
                ssize, spr_obits, eid, name, config, rtjs = processed_data[cfg_hash]
                s_computed = computed_bools[cfg_hash]

                new_bblocks = set(ggen.get_booltest_blocks(spr_obits))
                to_comp = new_bblocks.difference(s_computed)
                if len(to_comp) == 0:
                    continue

                config_obj = ggen.get_rtt_config_file(ssize)
                booltest_cfg = ggen.booltest_rtt_config(sorted(list(to_comp)))
                config_obj = ggen.update_booltest_config(config_obj, booltest_cfg)
                logger.info(f'To compute more for {eid} {name}, block lens {to_comp}, size {ssize}')

                nname = name.replace('.json', '') + '-boolex-' + '-'.join(map(str, to_comp))
                agg_scripts.append(
                    ExpRec(ename=nname,
                           ssize=ssize / 1024 / 1024, fname=nname + '.json',
                           tpl_file=json.loads(config), cfg_type='rtt-data-gen-config',
                           cfg_obj=config_obj, batteries=ggen.RttBatteries.BOOLTEST_2,
                           exp_data_size=ssize)
                )

            logger.info(f'Writing submit file, len: {len(agg_scripts)}')
            agg_filtered = []
            with open(os.path.join(tmpdir, '__enqueue.sh'), 'w+') as fh:
                fh.write('#!/bin/bash\n')
                for crec in agg_scripts:
                    name_find = self._name_find(crec.ename)
                    if name_find in existing_exps:
                        continue

                    logger.info(f'submit: {crec.ename}, {crec.fname}, {crec.ssize} MB')
                    agg_filtered.append(crec)

            logger.info(f'Submit size: {len(agg_filtered)}')
            write_submit_obj(agg_filtered, sdir=tmpdir)
            logger.info('Finished')

    def fix_booltest_runs(self, from_id=None):
        """Fixes Booltest runs with 0 total tests"""
        with self.conn.cursor() as c:
            logger.info("Processing experiments")

            c.execute("""SELECT j.id, j.battery, e.id, e.name, b.id, b.name, b.total_tests
                            FROM jobs j
                            JOIN experiments e on e.id = j.experiment_id
                            LEFT JOIN batteries b ON b.job_id = j.id
                            WHERE e.id >= %s AND j.status = 'finished' AND j.battery LIKE 'booltest%%' 
                            AND (b.total_tests is NULL OR b.total_tests = 0)
                            ORDER BY e.id
                              """ % (from_id or self.exp_id_low,))

            for result in c.fetchall():
                jid, jbat, eid, ename, bid, bname, btotal = result[0:7]

                if btotal == 0:
                    sql_exps = 'DELETE FROM batteries WHERE id=%s'
                    try_execute(lambda: c.execute(sql_exps, (bid,)),
                                msg="Delete battery with ID %s" % bid)

                if btotal is not None and btotal > 0:
                    continue

                sql_exps = 'UPDATE jobs SET status="pending" WHERE id=%s'
                try_execute(lambda: c.execute(sql_exps, (jid,)),
                            msg="Update job with ID %s" % jid)

                sql_exps = 'UPDATE experiments SET status="pending" WHERE id=%s'
                try_execute(lambda: c.execute(sql_exps, (eid,)),
                            msg="Update experiment with ID %s" % eid)

        self.conn.commit()

    def fix_incorrect_sizes(self, from_id=None, dry_run=False):
        """Delete all computed data that have less data computed """
        with self.conn.cursor() as c:
            logger.info("Processing experiments")

            c.execute("""SELECT e.id AS eid, e.name AS ename, e.data_file_size, e.expected_data_file_size, e.run_started
                            FROM experiments e 
                            WHERE e.id >= %s AND e.status = 'finished' AND
                             e.expected_data_file_size IS NOT NULL AND e.data_file_size IS NOT NULL AND
                             e.data_file_size != e.expected_data_file_size
                            ORDER BY e.id DESC
                              """ % (from_id or self.exp_id_low,))

            for result in c.fetchall():
                eid, ename, fsize, exp_fsize, tstart = result[0], result[1], int(result[2]), int(result[3]), result[4]
                diff = abs(exp_fsize - fsize)
                rtio = diff / float(exp_fsize)
                if fsize == exp_fsize:
                    continue

                if rtio <= 0.05:
                    # logger.info(f'File size mismatch, tolerable: {fsize} vs exp {exp_fsize}, diff {rtio}, '
                    #             f'eid {eid} {ename}')
                    continue

                logger.info(f'File size mismatch, {fsize} vs exp {exp_fsize}, diff {rtio}, eid {eid} {ename} {tstart}')
                if dry_run:
                    continue

                self._recompute_eid(c, eid=eid)
                self.conn.commit()

    def fix_broken_batteries(self, from_id=None, dry_run=False):
        """Removes all battery results that ended with DB errors, reschedules jobs and experiments to run"""
        with self.conn.cursor() as c:
            logger.info("Processing experiments")

            c.execute("""SELECT be.message, b.id AS bid, j.id AS jid, e.id AS eid, b.total_tests, b.name, e.name
                             FROM battery_errors be 
                             JOIN batteries b ON b.id = be.battery_id
                             JOIN jobs j ON b.job_id = j.id
                             JOIN experiments e ON e.id = b.experiment_id
                             WHERE e.id >= %s AND b.total_tests = 0 
                                AND be.message NOT LIKE '%%no tests were%%' 
                                AND (be.message LIKE 'Lock wait timeout%%' OR be.message LIKE 'Deadlock found%%' OR be.message LIKE '%%empty statistics%%')
                                AND e.name NOT LIKE '%%-Vision%%' 
                             ORDER BY b.id DESC     
                              """ % (from_id or self.exp_id_low,))

            for result in c.fetchall():
                msg, bid, jid, eid, totals, bname, ename = \
                    result[0], int(result[1]), int(result[2]), int(result[3]), int(result[4]), result[5], result[6]

                logger.info(f'Zero tests for battery, {eid} {ename} : {bname}, {msg}')
                if dry_run:
                    continue

                self._remove_bat(c, bid=bid, jid=jid, eid=eid)
                self.conn.commit()

    def fix_dup_ref(self, from_id=None, dry_run=False):
        """Removes duplicate reference runs"""
        with self.conn.cursor() as c:
            logger.info("Processing experiments")

            c.execute("""SELECT e.id AS eid, e.name, e.data_file_sha256
                             FROM experiments e
                             WHERE e.id >= %s  
                                AND e.name LIKE 'PH-SMREF%%' 
                             ORDER BY e.name DESC, e.id ASC
                              """ % (from_id or self.exp_id_low,))

            data = []
            for result in c.fetchall():
                eid, ename, ehash = \
                    int(result[0]), result[1], result[2]
                data.append((eid, ename, ehash))

            for k, g in itertools.groupby(data, lambda x: x[1]):
                g = list(g)
                if len(g) <= 1:
                    continue

                eid, ename, ehash = g[0]
                hashes = set([x[2] for x in g])
                logger.info(f'Duplicate ref, {eid} {ename} {ehash}, len: {len(g)}, hashes: {len(hashes)}')
                if len(hashes) != 1:
                    continue

                if dry_run:
                    continue

                for dels in g[1:]:
                    ceid = dels[0]
                    logger.info(f'Deleting experiment {ceid}, {dels}')
                    sql_exps = 'DELETE FROM experiments WHERE id=%s'
                    try_execute(lambda: c.execute(sql_exps, (ceid,)), msg="Delete experiments results for ID %s" % ceid)

                self.conn.commit()

    def recomp_experiments(self, experiments):
        """Delete all computation results and set jobs to recompute"""
        with self.conn.cursor() as c:
            logger.info("Processing experiments")
            for eid in experiments:
                logger.info(f'Deleting results for {eid}')

                self._recompute_eid(c, eid=eid)
                self.conn.commit()

    def fix_incomple_booltests(self, from_id=None, dry_run=False):
        """Recompute booltests with incomplete result sets"""
        with self.conn.cursor() as c:
            logger.info("Processing experiments")
            c.execute("""SELECT b.id AS bid, j.id AS jid, e.id AS eid, b.total_tests, b.name, e.name, j.run_started, data_file_sha256
                         FROM experiments e
                         JOIN batteries b ON b.experiment_id = e.id
                         JOIN jobs j ON b.job_id = j.id
                         WHERE e.id > %s 
                          AND b.total_tests < 36 and b.name LIKE 'booltest%%' AND j.retries < 10
                          AND (e.name LIKE 'PH%%' OR e.name LIKE 'testmpc%%') AND e.name NOT LIKE '%%Vision%%'
                         ORDER BY e.id DESC
                         LIMIT 1000
                        """ % (from_id or self.exp_id_low,))

            for result in c.fetchall():
                bid, jid, eid, total, bname, ename, jstarted, ehash = \
                    int(result[0]), int(result[1]), int(result[2]), int(result[3]), result[4], result[5], result[6], \
                    result[7]

                logger.info(f'Deleting results for {eid}, bid {bid}, jid {jid}, {ename} - {bname} total {total}, {jstarted}, {ehash}')

                if dry_run:
                    continue

                self._remove_bat(c, eid=eid, jid=jid, bid=bid)
                self.conn.commit()

    def fix_dup_results(self, from_id=None, dry_run=False):
        """Removes duplicate battery results"""
        with self.conn.cursor() as c:
            logger.info("Processing experiments")
            c.execute("""SELECT 
                        e.id AS eid, e.name AS ename
                        , b.id as bid, b.name as bname, b.total_tests, b.passed_tests
                        , b2.id as b2id, b2.total_tests, b2.passed_tests
                            FROM experiments e
                            LEFT JOIN batteries b ON b.experiment_id = e.id
                            LEFT JOIN batteries b2 ON b2.experiment_id = e.id AND b.id < b2.id AND b.name = b2.name
                            WHERE b.id is not null and b2.id is not null
                              AND e.id > %s AND (e.name LIKE 'PH%%' or e.name LIKE 'testmpc%%')
                            ORDER BY e.id DESC 
                        """ % (from_id or self.exp_id_low,))

            for result in c.fetchall():
                eid, ename, bid, bname, btotal, bpass, b2id, b2total, b2pass = \
                    int(result[0]), result[1], int(result[2]), result[3], int(result[4]), int(result[5]), \
                    int(result[6]), int(result[7]), int(result[8])

                logger.info(f'Duplicate results for {eid}: {ename}, {bid} {bname}, {bpass}/{btotal} vs {b2id} {b2pass}/{b2total}')
                if btotal != b2total:
                    logger.warning(f'Total num of tests mismatch')
                    continue

                if bpass != b2pass:
                    logger.warning(f'Passed num of tests mismatch')
                    continue

                logger.info(f'Removing {b2id}')
                if dry_run:
                    continue

                sql_exps = 'DELETE FROM batteries WHERE id=%s'
                try_execute(lambda: c.execute(sql_exps, (b2id,)), msg="Delete battery results for ID %s" % b2id)
                self.conn.commit()

    def _remove_bat(self, c, bid, jid, eid):
        sql_exps = 'DELETE FROM batteries WHERE id=%s'
        try_execute(lambda: c.execute(sql_exps, (bid,)), msg="Delete battery results for ID %s" % bid)

        sql_exps = 'UPDATE experiments SET status="running" WHERE id=%s'
        try_execute(lambda: c.execute(sql_exps, (eid,)), msg="Experiment reset for ID %s" % eid)

        sql_exps = 'UPDATE jobs SET status="pending", run_finished=NULL, run_heartbeat=NULL, ' \
                   'retries=0 WHERE id=%s'
        try_execute(lambda: c.execute(sql_exps, (jid,)), msg="Jobs reset for ID %s" % jid)

    def _recompute_eid(self, c, eid):
        sql_exps = 'DELETE FROM batteries WHERE experiment_id=%s'
        try_execute(lambda: c.execute(sql_exps, (eid,)), msg="Delete battery results for ID %s" % eid)

        sql_exps = 'UPDATE experiments SET status="pending", data_file_sha256=NULL, data_file_size=NULL WHERE id=%s'
        try_execute(lambda: c.execute(sql_exps, (eid,)), msg="Experiment reset for ID %s" % eid)

        sql_exps = 'UPDATE jobs SET status="pending", run_finished=NULL, run_heartbeat=NULL, ' \
                   'retries=0 WHERE experiment_id=%s'
        try_execute(lambda: c.execute(sql_exps, (eid,)), msg="Jobs reset for ID %s" % eid)

    def fix_boolex(self, from_id=None, dry_run=False):
        """Merge boolex results to appropriate testing battery"""
        with self.conn.cursor() as c:
            logger.info("Processing experiments")

            c.execute("""SELECT j.id, j.battery AS jbat, e.id AS eid, e.name AS ename, b.id AS bid, b.name AS bname, 
                            b.total_tests, b.passed_tests, b.pvalue,
                            e2.id AS e2id, e2.name AS e2name, b2.id as b2id,
                            b2.total_tests, b2.passed_tests, b2.pvalue
                            FROM jobs j
                            JOIN experiments e on e.id = j.experiment_id
                            JOIN rtt_data_providers dp ON e.data_file_id = dp.id
                            JOIN rtt_config_files dc ON e.config_file_id = dc.id
                            LEFT JOIN batteries b ON b.job_id = j.id
                            LEFT JOIN experiments e2 ON e2.data_file_id = e.data_file_id AND e2.id != e.id
                            LEFT JOIN batteries b2 ON b2.experiment_id = e2.id AND b2.name = b.name AND b2.id != b.id 
                            WHERE e.id >= %s AND e.status = 'finished' AND e.name LIKE '%%-boolex-%%' 
                            ORDER BY j.id DESC
                              """ % (from_id or self.exp_id_low,))

            for result in c.fetchall():
                jid, jbat, eid, ename, bid, bname, btotal, bpass, bpval = result[0:9]
                e2id, e2name = result[9:11]
                b2id, b2total, b2pass, b2pval = result[11:15]
                orig_ename = re.sub(r'-boolex-.*$', '', ename)
                e2name_cln = re.sub(r'\.json$', '', e2name)

                if b2id is None or e2id is None or bid is None or eid is None:
                    continue

                if orig_ename != e2name_cln:
                    logger.warning(f'Suspicious boolex record eid:jid:bid {eid}:{jid}:{bid} '
                                   f'connecting to eid2:bid2 {e2id}:{b2id}, '
                                   f'ename {ename} vs {e2name}')
                    continue

                npval = merge_pvals([bpval, b2pval])[0]
                ntotal = btotal + b2total
                npass = bpass + b2pass

                logger.info(f'eid:jid:bid {eid}:{jid}:{bid} connecting to eid2:bid2 {e2id}:{b2id}, '
                            f'ename {ename} vs {e2name}, '
                            f'total:pass:pval {bpval}:{btotal}:{bpass}, new '
                            f'total:pass:pval {b2pval}:{b2total}:{b2pass} result '
                            f'total:pass:pval {npval}:{ntotal}:{npass} '
                            )
                if dry_run:
                    continue

                print(f'Changing battery result to old experiment {bid} -> {b2id} (old)')
                csql = 'UPDATE tests SET battery_id=%s where battery_id=%s'
                try_execute(lambda: c.execute(csql, (b2id, bid,)),
                            msg="Changing battery result to old experiment")

                print(f'Update summary stats for {b2id} (old)')
                csql = 'UPDATE batteries SET total_tests=%s, passed_tests=%s, pvalue=%s where id=%s'
                try_execute(lambda: c.execute(csql, (ntotal, npass, npval, b2id,)),
                            msg="Update summary stats")

                print(f'Delete old battery info')
                csql = 'DELETE FROM batteries WHERE id=%s'
                try_execute(lambda: c.execute(csql, (bid,)),
                            msg="'Delete old battery info")
                self.conn.commit()

                print(f'Delete old experiment info')
                csql = 'DELETE FROM experiments WHERE id=%s'
                try_execute(lambda: c.execute(csql, (eid,)),
                            msg="Delete old experiment info")
                self.conn.commit()

            logger.info(f'Done')

    def fix_tangle(self):
        """Tangle experiments were mislabeled, just fix the name"""
        with self.conn.cursor() as c:
            logger.info("Processing experiments")

            c.execute("""SELECT e.id, name
                            FROM experiments e
                            JOIN rtt_data_providers dp ON e.data_file_id = dp.id
                            WHERE e.id >= %s and name LIKE '%%Tangle%%ctr.key%%'
                              """ % (self.exp_id_low,))

            renames = []
            for result in c.fetchall():
                eid, name = result[0], result[1]
                renames.append((eid, name, name.replace('ctr.key', 'ctr')))

            for eid, oname, nname in renames:
                sql_exps = 'UPDATE experiments SET name=%s WHERE id=%s'
                print(f'Updating {eid} to {nname}')
                try_execute(lambda: c.execute(sql_exps, (nname, eid,)),
                            msg="Update experiment with ID %s" % eid)

            self.conn.commit()

    def outphase_old_prngs(self, from_id=None):
        """Remove old PRNGs experiments"""
        with self.conn.cursor() as c:
            logger.info("Processing experiments")
            sql_sel = """SELECT e.id, name, dp.`provider_config`, dp.provider_name, dp.id, dp.provider_config_name
                            FROM experiments e
                            JOIN rtt_data_providers dp ON e.data_file_id = dp.id
                            WHERE e.id >= %s AND e.id < 339060
                            AND name LIKE 'PH4-SM%%' AND name LIKE '%%t:prng%%'                            
                      """ % (from_id,)
            logger.info('SQL: %s' % sql_sel)
            c.execute(sql_sel)

            renames = []
            for result in c.fetchall():
                eid, name = result[0], result[1]
                renames.append((eid, name))

            for eid, name in renames:
                nname = 'SUSP-%s' % name

                sql_exps = 'UPDATE experiments SET name=%s WHERE id=%s'
                print(f'Updating {eid} to {nname}')
                try_execute(lambda: c.execute(sql_exps, (nname, eid,)),
                            msg="Update experiment with ID %s" % eid)
            self.conn.commit()

    def fix_init_freq(self, from_id=None):
        """Some experiments did not have init_frequency field, has to be added"""
        with self.conn.cursor() as c:
            logger.info("Processing experiments")
            sql_sel = """SELECT e.id, name, dp.`provider_config`, dp.provider_name, dp.id, dp.provider_config_name
                            FROM experiments e
                            JOIN rtt_data_providers dp ON e.data_file_id = dp.id
                            JOIN jobs j ON j.experiment_id = e.id 
                            WHERE e.id >= %s 
                            AND name LIKE 'PH4-SM%%'
                            AND j.status = 'running'
                            GROUP BY e.id
                      """ % (from_id,)
            logger.info('SQL: %s' % sql_sel)
            c.execute(sql_sel)

            reconfigured = {}
            for result in c.fetchall():
                eid, name, config, pname, pid, ppname = result[0], result[1], result[2], result[3], result[4], result[5]
                config_js = json.loads(config)

                if 'init_frequency' in config_js['stream']:
                    continue
                if config_js['stream']['type'] != 'block':
                    continue

                if pid not in reconfigured:
                    config_js['stream']['init_frequency'] = 'only_once'
                    js_str = json.dumps(config_js, indent=2)
                    js_hash = hashlib.sha256(js_str.encode('utf8')).hexdigest()

                    npid = None
                    with self.conn.cursor() as c2:
                        sql_sel_dp = "SELECT id FROM rtt_data_providers WHERE provider_config_hash=%s"
                        c2.execute(sql_sel_dp, (js_hash,))
                        cr = c2.fetchall()
                        if cr and len(cr) > 0:
                            npid = cr[0][0]

                    if npid is None:
                        ql_ins_csc = "INSERT INTO rtt_data_providers(provider_name, provider_config, provider_config_name, " \
                                     "provider_config_hash) " \
                                     "VALUES(%s,%s,%s,%s)"
                        try_execute(lambda: c.execute(ql_ins_csc, (pname, js_str, ppname, js_hash)))
                        self.conn.commit()

                        if c.lastrowid is None or c.lastrowid == 0:
                            raise ValueError(f'Could not get inserted ID for hash {js_hash}')
                        npid = c.lastrowid

                    reconfigured[pid] = npid
                    if 'broken-' not in ppname:
                        sql_update_old = 'UPDATE rtt_data_providers SET provider_config_name=%s WHERE id=%s'
                        try_execute(lambda: c.execute(sql_update_old, (f'broken-{ppname}', pid)))

                sql_exps = 'UPDATE experiments SET data_file_id=%s WHERE id=%s'
                logger.info(f'Updating reconfigured PID {pid} to {reconfigured[pid]} for eid {eid}')
                try:
                    c.execute(sql_exps, (reconfigured[pid], eid,))
                except Exception as e:
                    logger.warning(f'Could not update, skipping {eid}')
                    continue

                # try_execute(lambda: c.execute(sql_exps, (reconfigured[pid], eid,)),
                #             msg="Update experiment with ID %s" % eid)
                self.conn.commit()

            logger.info('Finished')

    def fix_1000MB(self, from_id=None, exp_count=3):
        """1000MB experiments could have HW configuration that was overlaping / overflowing
        We have to recompute HW weight so it can cover the whole interval without overlaps"""

        with self.conn.cursor() as c:
            bad_hws = []
            redundant_hws = []

            logger.info("Processing experiments")
            sql_sel = """SELECT e.id, name, dp.`provider_config`, dp.provider_name, dp.id, dp.provider_config_name
                                        FROM experiments e
                                        JOIN rtt_data_providers dp ON e.data_file_id = dp.id
                                        WHERE e.id > %s 
                                        AND name LIKE 'PH4-SM%%'
                                        AND name LIKE '%%-i:hw%%'
                      """ % (from_id,)
            logger.info('SQL: %s' % sql_sel)
            c.execute(sql_sel)

            for result in c.fetchall():
                eid, name, config, pname = result[0], result[1], result[2], result[3]
                config_js = json.loads(config)

                alg_type = config_js['stream']['type']
                if '-i:hw.key-' in name:
                    inpkey = 'key'
                    isize = config_js['stream']['key_size']

                else:
                    inpkey = get_input_key(alg_type)
                    isize = get_input_size(config_js)

                new_size = config_js['tv_count'] * config_js['tv_size']
                cur_hw = config_js['stream'][inpkey]
                tv_count = config_js['tv_count']
                m1 = re.match(r'.*-e:([\d]+)-.*', name)
                lhw_offset_ratio = 0
                eidx = 0
                if m1:
                    eidx = int(m1.group(1))
                    lhw_offset_ratio = eidx / exp_count
                else:
                    logger.info(f'Could not detect current HW offset ratio, using 0')

                data_size_in = int(math.ceil((new_size / tv_count) * isize))
                try:
                    weight_comp = comp_hw_weight(isize, exp_count, min_samples=tv_count)
                except:
                    logger.info(f'! HW not possible for {eid} {name}, tv_count={tv_count}, isize: {isize}')
                    continue

                # Check current config, if the HW setting is OK enough, no need to discard.
                # Criteria: HW spaces do not overlap. Assuming equal spacing,
                # remaining vectors >= tv_count * remaining offsets. We cannot check offset 0 easily.
                cur_num_vectors = comb_cached(isize * 8, cur_hw['hw'])
                offset_idx = rank(cur_hw['initial_state'], isize * 8)
                rem_vectors = cur_num_vectors - offset_idx
                num_space_to_cover = tv_count * (exp_count - 1 - eidx)

                if cur_hw['hw'] == weight_comp and rem_vectors >= num_space_to_cover:
                    # logger.info(f'HW config is OK for {name} [1]')
                    continue

                if cur_hw['hw'] >= weight_comp and rem_vectors >= num_space_to_cover:
                    logger.info(f'HW config is OK but redundant for {name} [1], usedhw {cur_hw["hw"]} vs comp {weight_comp}, isize {isize}')
                    redundant_hws.append((eid, name))
                    continue

                try:
                    hw_cfg = make_hw_config(isize, weight=weight_comp, offset_range=lhw_offset_ratio,
                                            min_data=data_size_in, return_aux=True)  # type: HwConfig

                    if weight_comp == cur_hw['hw'] and hw_cfg.core['initial_state'] == cur_hw['initial_state']:
                        logger.info(f'HW config is OK for {name} [2]')
                        continue

                    logger.info(f'!HW config not enough for {name}, new config: {hw_cfg.note}, {hw_cfg.gen_data_mb} {hw_cfg.rem_vectors}')
                    bad_hws.append((eid, name))
                    continue

                except Exception as e:
                    logger.error(f"Exception in hw gen {e}, alg: {name}, "
                                 f"wcomp: {weight_comp}, tv-count: {tv_count}, ilen: {isize}, "
                                 f"dsizeInp: {data_size_in}", exc_info=e)
                    raise

            logger.info('Redundant HWs: %s' % (', '.join([str(x[0]) for x in redundant_hws])))
            for eid, name in redundant_hws:
                logger.info(f'  {eid} {name}')

            logger.info('Invalid HWs: %s' % (', '.join([str(x[0]) for x in bad_hws])))
            for eid, name in bad_hws:
                logger.info(f'  {eid} {name}')

            logger.info('Finished')

    def dump_experiment_configs(self, from_id: int, tmpdir='/tmp/rspecs', matcher=None, clean_before=False):
        """
        Loads all experiment configuration files and dumps them as jsons to the file system.
        Used by analysis scripts to resubmit work.
        """
        if clean_before and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

        os.makedirs(tmpdir, exist_ok=True)
        expfiles = set()
        with self.conn.cursor() as c:
            logger.info("Loading experiment input configs: %s" % (from_id,))
            sql_sel = """SELECT e.id, name, dp.`provider_config`, dp.provider_name, dp.id, dp.provider_config_name
                            FROM experiments e
                            JOIN rtt_data_providers dp ON e.data_file_id = dp.id
                            WHERE e.id >= %s
                              """ % (from_id,)
            logger.info('SQL: %s' % sql_sel)
            c.execute(sql_sel)

            for result in c.fetchall():
                eid, name, config, pname = result[0], result[1], result[2], result[3]
                if matcher and not matcher((eid, name, config, pname)):
                    continue

                js = json.loads(config)
                if 'note' in js:
                    js['note'] = name
                    config = json.dumps(js, indent=2)

                elif 'input_files' in js and 'stream' in js and 'exec' in js['stream']:
                    js['stream']['note'] = name
                    config = json.dumps(js, indent=2)

                fname = name.replace(':', '_').replace('.json', '') + '.json'
                fpath = os.path.join(tmpdir, fname)
                with open(fpath, 'w+') as fh:
                    fh.write(config)

                expfiles.add(fname)

        with open(os.path.join(tmpdir, '__expfiles.json'), 'w+') as fh:
            js = list(expfiles)
            js.sort(key=natural_sort_key)
            json.dump(js, fh)

    def _name_find(self, nname_find):
        nname_find = re.sub(r'^PH4?-SM-([\d]+)-', '', nname_find)
        nname_find = re.sub(r'^testmpc([\d]+)-', '', nname_find)
        nname_find = re.sub(r'-i:(.+?)-.*$', '-i:\\1', nname_find)  # drop inp detailed specs
        return nname_find

    def _load_existing_exps(self, c, skip_existing_since):
        existing_exps = {}
        if skip_existing_since is None:
            return existing_exps

        logger.info("Loading exp name database to skip existing")
        sql_sel = """SELECT e.id, name FROM experiments e
                                     WHERE e.id >= %s
                                  """ % (skip_existing_since,)
        c.execute(sql_sel)
        for result in c.fetchall():
            nname_find = self._name_find(result[1])
            existing_exps[nname_find] = result[0]
        logger.info('Loaded database of %s expnames; %s' % (len(existing_exps), sql_sel,))
        return existing_exps

    def comp_new_rounds(self, specs, tmpdir='/tmp/rspecs', smidx=5, new_size=None, skip_existing_since=None,
                        clean_before=False):
        """
        Generates submit_experiment for a new rounds to compute.
        specs is: fname -> meth -> [[exids], [rounds]]

        Old method, uses exids to generate job data provider configuration, just increasing the rounds
        """
        if clean_before and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

        os.makedirs(tmpdir, exist_ok=True)

        eids_map = collections.defaultdict(lambda: set())
        for fname in specs:
            for meth in specs[fname]:
                rec = specs[fname][meth]
                for cexid in rec[0]:
                    for cround in rec[1]:
                        eids_map[cexid].add(cround)

        file_names = []
        with self.conn.cursor() as c:
            existing_exps = self._load_existing_exps(c, skip_existing_since)

            logger.info("Generating new round specs, eids: %s" % (', '.join(map(str, list(eids_map.keys())))))
            sql_sel = """SELECT e.id, name, dp.`provider_config`, dp.provider_name, dp.id, dp.provider_config_name
                            FROM experiments e
                            JOIN rtt_data_providers dp ON e.data_file_id = dp.id
                            WHERE e.id IN (%s)
                              """ % (','.join(['%s' for _ in eids_map]),)
            logger.info('SQL: %s' % sql_sel)

            c.execute(sql_sel, list(eids_map.keys()))

            for result in c.fetchall():
                eid, name, config, pname = result[0], result[1], result[2], result[3]
                config_js = json.loads(config)

                if pname == 'cryptostreams':
                    for cround in sorted(list(eids_map[eid])):
                        try:
                            nname, ssize = Cleaner.change_cstreams(config_js=config_js, cround=cround,
                                                                   name=name, smidx=smidx, new_size=new_size)
                        except HwSpaceTooSmall as e:
                            logger.warning(f'HW space too small for {eid} {name}')
                            continue

                        ptype = '--cryptostreams-config'
                        fname = nname.replace(':', '_').replace('.json', '') + '.json'
                        nname_find = re.sub(r'^PH4-SM-([\d]+)-', '', nname)
                        if nname_find in existing_exps:
                            logger.info('  Eid %s -> r=%s, name=%s, fname=%s, nsize=%s SKIP'
                                        % (eid, cround, nname, fname, new_size))
                            continue

                        logger.info('  Eid %s -> r=%s, name=%s, fname=%s, nsize=%s'
                                    % (eid, cround, nname, fname, new_size))

                        file_names.append((nname, fname, ssize, ptype))
                        with open(os.path.join(tmpdir, fname), 'w+') as fh:
                            json.dump(config_js, fh, indent=2)

                elif pname == 'rtt-data-gen':
                    logger.info('RTT data gen for %s' % (name,))
                    if 'stream' not in config_js:
                        logger.info(f'stream key not found for {name}')
                        continue
                    if 'exec' not in config_js['stream']:
                        logger.info(f'stream.exec key not found for {name}')
                        continue
                    if 'input_files' not in config_js:
                        logger.info(f'input_files key not found for {name}')
                        continue
                    if 'CONFIG1.JSON' not in config_js['input_files']:
                        logger.info(f'input_files.CONFIG1.JSON key not found for {name}')
                        continue
                    if 'data' not in config_js['input_files']['CONFIG1.JSON']:
                        logger.info(f'input_files.`CONFIG1.JSON`.data key not found for {name}')
                        continue
                    if 'testmpc' not in name:
                        logger.info(f'Only testmpc experiments are supported yet. Skipping {name}')
                        continue

                    if '{RTT_EXEC}' in config_js['stream']['exec']:
                        # Size definition is more complex for MPC algorithms as we have different
                        # sizes of input output blocks. This should be handled by MPC generator.
                        # TODO: use generator to regenerate these, do not transform config files.
                        logger.info(f'Rtt-exec generated streams not supported at the moment. Skipping {name}')
                        continue

                    ptype = '--rtt-data-gen-config'
                    dfile = config_js['input_files']['CONFIG1.JSON']['data']

                    # Works for lowmc, functions generated by python have -r rnd in `exec`
                    try:
                        _, ssize = Cleaner.change_cstreams(config_js=dfile, cround=cround,
                                                           name=name, smidx=smidx, new_size=new_size)
                    except HwSpaceTooSmall as e:
                        logger.warning(f'HW space too small for {eid} {name}')
                        continue

                    # Adapt round and size in the exp name
                    # Name example: testmpc06-lowmc-s128d-bin-raw-r65-inp-lhw02-b16-w5-spr--s100MB.json
                    nname = re.sub(r'-r([\d]+)-', '-r%s-' % cround, name)
                    nname = re.sub(r'^testmpc([\d]+)-', 'testmpc%02d-' % smidx, nname)
                    ssize = '100' if ('-s100MiB' in name or '-s100MB' in name) else '10'
                    if new_size:
                        nname = re.sub(r'-s([\d]+)Mi?B\b', '-s%sMB' % (int(new_size / 1024 / 1024)), nname)
                        ssize = str(int(new_size / 1024 / 1024))

                    config_js['note'] = nname

                    # Fname conversion, dump
                    fname = nname.replace(':', '_').replace('.json', '') + '.json'
                    nname_find = self._name_find(nname)
                    if nname_find in existing_exps:
                        logger.info('  Eid %s -> r=%s, name=%s, fname=%s, nsize=%s SKIP'
                                    % (eid, cround, nname, fname, new_size))
                        continue

                    logger.info('  Eid %s -> r=%s, name=%s, fname=%s, nsize=%s'
                                % (eid, cround, nname, fname, new_size))

                    file_names.append((nname, fname, ssize, ptype))
                    with open(os.path.join(tmpdir, fname), 'w+') as fh:
                        json.dump(config_js, fh, indent=2)

                else:
                    print('Could not process provider %s' % (pname,))
                    continue

        logger.info('Writing submit file')
        with open(os.path.join(tmpdir, '__enqueue.sh'), 'w+') as fh:
            fh.write('#!/bin/bash\n')
            for ix, (nname, fname, ssize, ptype) in enumerate(sorted(file_names, key=lambda x: x[0])):
                logger.info(f'submit: {nname}, {fname}, {ssize} MB')
                fh.write(f"echo '{ix}/{len(file_names)}'\n")
                fh.write(f"submit_experiment --all_batteries --name '{nname}' --cfg '/home/debian/rtt-home/RTTWebInterface/media/predefined_configurations/{ssize}MB.json' {ptype} '{fname}'\n")

    def comp_new_rounds_new(self, specs, tmpdir='/tmp/rspecs', smidx=5, skip_existing_since=None, new_size=None,
                            clean_before=False, randomize_seed=False, skip_mpc=False, skip_large=False):
        """
        Generates submit_experiment for a new rounds to compute.
        specs is: ftype:fname -> meth:size -> [rounds]

        New method, creates data provider configuration from the scratch using specs and experiment generator.
        """
        if clean_before and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        os.makedirs(tmpdir, exist_ok=True)

        eprefix = 'PH4-SM-%02d-' % smidx
        eprefix_mpc = 'testmpc%02d-' % smidx
        agg_scripts = []  # type: list[ExpRec]

        num_skip_already_done = 0
        num_skip_size = 0
        num_skip_mpc = 0
        with self.conn.cursor() as c:
            existing_exps = self._load_existing_exps(c, skip_existing_since)
            for ftypename in specs.keys():
                fparts = ftypename.split(':')
                ftype, funcname = int(fparts[0]), fparts[1]

                erec = FUNC_DB.search(funcname, ftype)
                if erec is None:
                    logger.info(f'Function {ftypename} not found')
                    continue

                if erec.stype == FuncInfo.MPC:
                    if skip_mpc:
                        num_skip_mpc += 1
                        continue
                    mpc_res = self.comp_mpc(erec, ftypename, specs[ftypename], eprefix_mpc, randomize_seed)  # type: Optional[List[ExpRec]]
                    if mpc_res:
                        for x in mpc_res:
                            x.priority = 50
                        agg_scripts += mpc_res
                    continue

                try:
                    alg_type = erec.get_alg_type()
                except:
                    logger.info(f'Skipping unsupported {ftypename}, specs: {specs[ftypename]}')
                    continue

                if alg_type == 'prng':
                    logger.info(f'Skipping unsupported PRNG {ftypename}, specs: {specs[ftypename]}')
                    continue

                new_size_used = set()
                for methsize in specs[ftypename].keys():
                    meth_parts = methsize.split(':')
                    if len(meth_parts) > 2:
                        logger.info(f'Skipping unsupported method {ftypename} for {methsize}')
                        continue

                    meth, size = meth_parts[0], int(meth_parts[-1])
                    if new_size is not None:
                        meth_base = ':'.join(meth_parts[:-1])
                        if meth_base in new_size_used:
                            continue
                        size = new_size
                        new_size_used.add(meth_base)

                    methsubs = meth.split('..')
                    has_key_prim = '.key' in methsubs[0]

                    prim_stream = StreamOptions.from_str(methsubs[0])  # key_stream for .key, plaintext for inp
                    has_plain_sec = len(methsubs) > 1 and '.inp' in methsubs[1]

                    sec_stream_type = methsubs[1].split('.')[0] if has_plain_sec else None
                    sec_stream_str = StreamOptions.from_str(sec_stream_type) if sec_stream_type else None
                    if sec_stream_str is None and has_key_prim:  # zero is default for col
                        sec_stream_str = StreamOptions.ZERO
                    if sec_stream_str is None and not has_key_prim:  # zero is default for default
                        sec_stream_str = StreamOptions.RND

                    rnd_info = specs[ftypename][methsize]
                    if isinstance(rnd_info, (list, tuple)) and len(rnd_info) == 2 and isinstance(rnd_info[1], (list, tuple)):
                        rnd_info = rnd_info[1]

                    for rnd in sorted(list(rnd_info)):
                        try:
                            if has_key_prim:
                                agg_scripts += generate_cfg_col(
                                    alg_type, funcname, size, cround=rnd,
                                    tv_size=erec.block_size, key_size=erec.key_size, iv_size=erec.iv_size,
                                    nexps=3, eprefix=eprefix,
                                    streams=prim_stream, inp_stream=sec_stream_str, randomize_seed=True)
                            else:
                                agg_scripts += generate_cfg_inp(
                                    alg_type, funcname, size, cround=rnd,
                                    tv_size=erec.block_size, key_size=erec.key_size, iv_size=erec.iv_size,
                                    nexps=3, eprefix=eprefix,
                                    streams=prim_stream, key_stream=sec_stream_str, randomize_seed=True)

                        except Exception as e:
                            logger.warning(f'Error in processing: {alg_type}:{funcname}:{size}:{rnd} {erec}, {methsize} err: {e}')
                            continue

        logger.info(f'Writing submit file, len: {len(agg_scripts)}')
        agg_filtered = []
        with open(os.path.join(tmpdir, '__enqueue.sh'), 'w+') as fh:
            fh.write('#!/bin/bash\n')
            for crec in agg_scripts:
                name_find = self._name_find(crec.ename)
                if name_find in existing_exps:
                    num_skip_already_done += 1
                    continue
                if skip_large and crec.ssize >= 1000:
                    num_skip_size += 1
                    continue
                if crec.ssize >= 1000:
                    crec.priority = 100

                logger.info(f'submit: {crec.ename}, {crec.fname}, {crec.ssize} MB')
                agg_filtered.append(crec)

        logger.info(f'Submit size: {len(agg_filtered)}, skip done: {num_skip_already_done}, skip size: {num_skip_size}'
                    f', skip mpc: {num_skip_mpc}')
        write_submit_obj(agg_filtered, sdir=tmpdir)

    def comp_mpc(self, erec: FuncInfo, ftypename: str, specs, eprefix=None, randomize_seed=False):
        if 'lowmc' in erec.fname:
            return self.comp_lowmc(erec, ftypename, specs, eprefix, randomize_seed)

        logger.info(f'Processing {ftypename}, specs: {specs}')
        agg_scripts = []
        fname = erec.fname

        for methsize in specs.keys():
            meth_parts = methsize.split(':')
            meth, spread, size = meth_parts[0], meth_parts[1], int(meth_parts[-1])
            methsubs = meth.split('..')
            has_key_prim = '.key' in methsubs[0]

            prim_stream = StreamOptions.from_str(methsubs[0])  # key_stream for .key, plaintext for inp
            has_plain_sec = len(methsubs) > 1 and '.inp' in methsubs[1]

            sec_stream_type = methsubs[1].split('.')[0] if has_plain_sec else None
            sec_stream_str = StreamOptions.from_str(sec_stream_type) if sec_stream_type else None
            if sec_stream_str is None and has_key_prim:  # zero is default for col
                sec_stream_str = StreamOptions.ZERO
            if sec_stream_str is None and not has_key_prim:  # zero is default for default
                sec_stream_str = StreamOptions.RND

            rnd_info = specs[methsize]
            if isinstance(rnd_info, (list, tuple)) and len(rnd_info) == 2 and isinstance(rnd_info[1], (list, tuple)):
                rnd_info = rnd_info[1]
            rnd_list = sorted(list(rnd_info))

            fname_params = fname.replace('gmimc-', '').replace('mimc_hash-', '')
            if fname_params not in MPC_SAGE_PARAMS:
                logger.error(f'Unknown MPC function {fname}, no parameters')
                continue

            params = MPC_SAGE_PARAMS[fname_params]
            to_gen_tpl = [fname_params, params.field, params.full_rounds, params.script, params.round_tpl,
                          [(x,) for x in rnd_list]]
            try:
                if fname.startswith('Poseidon'):
                    to_gen_tpl[-1] = [(r, 0, 0) for r in rnd_list]
                elif fname.startswith('Starkad'):
                    to_gen_tpl[-1] = [(r, 0, 0) for r in rnd_list]
                elif fname.startswith('Rescue'):
                    pass
                elif fname.startswith('Vision'):
                    pass
                elif fname.startswith('gmimc'):
                    pass
                elif fname.startswith('mimc'):
                    pass
                else:
                    logger.error(f'Unknown MPC function {fname}')
                    continue

                if has_key_prim:
                    logger.error(f'.key not supported for {fname}')
                    continue

                agg_scripts += gen_script_config(
                    [to_gen_tpl], params.is_prime(), data_sizes=[size],
                    eprefix=eprefix, streams=prim_stream,
                    use_as_key=has_key_prim, other_stream=sec_stream_str,
                    randomize_seed=randomize_seed, spread=spread)

            except Exception as e:
                logger.warning(
                    f'Error in processing: {erec.fname}:{size}:{rnd_info} {erec}, {methsize} err: {e}')
                continue
        return agg_scripts

    def comp_lowmc(self, erec: FuncInfo, ftypename: str, specs, eprefix=None, randomize_seed=False):
        logger.info(f'Processing {ftypename}, specs: {specs}')
        agg_scripts = []

        for methsize in specs.keys():
            meth_parts = methsize.split(':')
            meth, size = meth_parts[0], int(meth_parts[-1])
            methsubs = meth.split('..')
            has_key_prim = '.key' in methsubs[0]

            prim_stream = StreamOptions.from_str(methsubs[0])  # key_stream for .key, plaintext for inp
            has_plain_sec = len(methsubs) > 1 and '.inp' in methsubs[1]

            sec_stream_type = methsubs[1].split('.')[0] if has_plain_sec else None
            sec_stream_str = StreamOptions.from_str(sec_stream_type) if sec_stream_type else None
            if sec_stream_str is None and has_key_prim:  # zero is default for col
                sec_stream_str = StreamOptions.ZERO
            if sec_stream_str is None and not has_key_prim:  # zero is default for default
                sec_stream_str = StreamOptions.RND

            rnd_info = specs[methsize]
            if isinstance(rnd_info, (list, tuple)) and len(rnd_info) == 2 and isinstance(rnd_info[1], (list, tuple)):
                rnd_info = rnd_info[1]

            try:
                agg_scripts += gen_lowmc_core(
                    [(erec.fname, (None,), sorted(list(rnd_info)))], data_sizes=[size], eprefix=eprefix,
                    streams=prim_stream, use_as_key=has_key_prim,
                    other_stream=sec_stream_str, randomize_seed=randomize_seed)

            except Exception as e:
                logger.warning(
                    f'Error in processing: {erec.fname}:{size}:{rnd_info} {erec}, {methsize} err: {e}')
                continue
        return agg_scripts

    @staticmethod
    def change_cstreams(config_js, cround, name, smidx, new_size=None, exp_count=3):
        config_js['stream']['round'] = cround
        nname = re.sub(r'-r:([\d]+)-', '-r:%s-' % cround, name)
        nname = re.sub(r'^PH4-SM-([\d]+)-', 'PH4-SM-%02d-' % smidx, nname)

        ssize = '100' if ('-s:100MiB-' in name or '-s:100MB' in name) else '10'
        if new_size:
            nname = re.sub(r'-s:([\d]+)Mi?B\b', '-s:%sMiB' % (int(new_size / 1024 / 1024)), nname)
            tv_count = config_js['tv_count'] = int(math.ceil(new_size / config_js['tv_size']))
            ssize = str(int(new_size / 1024 / 1024))

            # LHW counter needs reoffsetting as the only input here
            alg_type = config_js['stream']['type']
            if '-i:hw.key-' in name:
                logger.info('Adjusting input HW for a key input')
                inpkey = 'key'
                isize = config_js['stream']['key_size']

            else:
                logger.info('Adjusting input HW for input')
                inpkey = get_input_key(alg_type)
                isize = get_input_size(config_js)

            cur_hw = config_js['stream'][inpkey]
            lhw_offset_ratio = cur_hw['offset_ratio'] if 'offset_ratio' in cur_hw else 0.0

            m1 = re.match(r'.*-e:([\d]+)-.*', name)
            m2 = re.match(r'-lhw([\d]+)-', name)
            if m1:
                lhw_offset_ratio = int(m1.group(1)) / exp_count
            elif m2:
                lhw_offset_ratio = int(m2.group(1)) / exp_count
            else:
                logger.info(f'Could not detect current HW offset ratio, using 0')

            data_size_in = int(math.ceil((new_size / tv_count) * isize))
            weight_comp = comp_hw_weight(isize, exp_count, min_samples=tv_count)
            try:
                hw_cfg = make_hw_config(isize, weight=weight_comp, offset_range=lhw_offset_ratio, min_data=data_size_in, return_aux=True)  # type: HwConfig
                config_js['stream'][inpkey] = hw_cfg.core
                nname = re.sub(r'(-i_hw(:?.key)?)-.*', f'\\1-{hw_cfg.note}', nname)

            except Exception as e:
                logger.error(f"Exception in hw gen {e}, alg: {name}, "
                             f"wcomp: {weight_comp}, tv-count: {tv_count}, ilen: {isize}, "
                             f"dsizeInp: {data_size_in}", exc_info=e)
                raise

        config_js['note'] = nname
        return nname, ssize


def main():
    l = Cleaner()
    l.main()


if __name__ == "__main__":
    main()
