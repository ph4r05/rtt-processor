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

from rtt_tools.generator_mpc import get_input_key, get_input_size, comp_hw_weight, make_hw_config, HwConfig, \
    comb_cached, rank, HwSpaceTooSmall, generate_cfg_col, StreamOptions, ExpRec, write_submit_obj, generate_cfg_inp
from rtt_tools.gen.max_rounds import FUNC_DB, FuncDb, FuncInfo
from rtt_tools.utils import natural_sort_key

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
                if config_js['stream']['type'] in ['hash', 'prng']:
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
        nname_find = re.sub(r'^PH4-SM-([\d]+)-', '', nname_find)
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
                            clean_before=False):
        """
        Generates submit_experiment for a new rounds to compute.
        specs is: ftype:fname -> meth:size -> [rounds]

        New method, creates data provider configuration from the scratch using specs and experiment generator.
        """
        if clean_before and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        os.makedirs(tmpdir, exist_ok=True)

        eprefix = 'PH4-SM-%02d-' % smidx
        agg_scripts = []  # type: list[ExpRec]
        with self.conn.cursor() as c:
            existing_exps = self._load_existing_exps(c, skip_existing_since)
            for ftypename in specs.keys():
                fparts = ftypename.split(':')
                ftype, funcname = int(fparts[0]), fparts[1]

                erec = FUNC_DB.search(funcname, ftype)
                if erec is None:
                    logger.info(f'Function {ftypename} not found')
                    continue

                try:
                    alg_type = erec.get_alg_type()
                except:
                    logger.info(f'Skipping unsupported {ftypename}')
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

                    sec_stream_type = methsubs[1].split('.')[1] if has_plain_sec else None
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
                                    streams=prim_stream, inp_stream=sec_stream_str)
                            else:
                                agg_scripts += generate_cfg_inp(
                                    alg_type, funcname, size, cround=rnd,
                                    tv_size=erec.block_size, key_size=erec.key_size, iv_size=erec.iv_size,
                                    nexps=3, eprefix=eprefix,
                                    streams=prim_stream, key_stream=sec_stream_str)

                        except Exception as e:
                            logger.warning(f'Error in processing: {alg_type}:{funcname}:{size}:{rnd} {erec}, {methsize} err: {e}')
                            continue

        logger.info('Writing submit file')
        agg_filtered = []
        with open(os.path.join(tmpdir, '__enqueue.sh'), 'w+') as fh:
            fh.write('#!/bin/bash\n')
            for crec in agg_scripts:
                name_find = self._name_find(crec.ename)
                if name_find in existing_exps:
                    continue

                logger.info(f'submit: {crec.ename}, {crec.fname}, {crec.ssize} MB')
                agg_filtered.append(crec)

        write_submit_obj(agg_filtered, sdir=tmpdir)

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
