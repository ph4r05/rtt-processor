#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import logging
import coloredlogs
import argparse
import glob
import os
import itertools
import functools
import json
import copy
import io
from collections import OrderedDict, namedtuple
from .ranking import rank, unrank, comb_cached
from .gentools import jsonpath, comp_hw_weight, log2ceil, make_hw_core, make_ctr_core, comp_hw_data

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class AlgRec:
    def __init__(self, alg, alg_type, alg_round, tv_size, js, ipath=None, paths=None):
        self.alg = alg
        self.alg_type = alg_type
        self.alg_round = alg_round
        self.tv_size = tv_size
        self.js = js
        self.ipath = ipath
        self.paths = paths or []

    def __eq__(self, o: object) -> bool:
        return (self.alg, self.alg_type, self.alg_round, self.tv_size) == (o.alg, o.alg_type, o.alg_round, o.tv_size) if isinstance(o, AlgRec) else super().__eq__(o)

    def __repr__(self) -> str:
        return f'Alg({self.alg}, {self.alg_type}, {self.alg_round}, {self.tv_size})'

    def __hash__(self) -> int:
        return hash((self.alg, self.alg_type, self.alg_round, self.tv_size))


class TaroProc:
    def __init__(self):
        self.args = None

    def arg_parse(self):
        parser = argparse.ArgumentParser(description='TaroProc')
        parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                            help='enables debug mode')

        parser.add_argument('dirs', nargs=argparse.ONE_OR_MORE)
        parser.add_argument('--ec', dest='ec', type=int, default=3,
                            help='Experiment total count')
        parser.add_argument('--out', dest='out',
                            help='Output directory')
        parser.add_argument('--prefix', dest='prefix',
                            help='Experiment prefix')
        self.args = parser.parse_args()

    def work(self):
        self.arg_parse()
        if self.args.debug:
            coloredlogs.install(level=logging.DEBUG, use_chroot=False)

        logger.info(f'Processing {self.args.dirs}')
        paths = functools.reduce(lambda x, y: x+y, [glob.glob('%s/*.json' % (d,)) for d in self.args.dirs], [])
        logger.info(f'Number of files found: {len(paths)}')

        SZ_10_MIB = 1024 * 1024 * 10
        SZ_100_MIB = 1024 * 1024 * 100
        eprefix = f'{self.args.prefix}-' if self.args.prefix else ''

        alg_acc = []
        alg_acc_s = {}

        # Aggregate
        for ipath in paths:
            with open(ipath) as fh:
                js = json.load(fh)

            stream = js['stream']
            alg = stream['algorithm']
            alg_type = stream['type']
            alg_round = jsonpath('$.round', stream, True)
            blen = js['tv_size']

            rec = AlgRec(alg=alg, alg_type=alg_type, alg_round=alg_round, tv_size=blen, js=js, ipath=ipath)
            if rec in alg_acc_s:
                alg_acc_s[rec].paths.append(ipath)
                continue

            alg_acc.append(rec)
            alg_acc_s[rec] = rec

        alg_acc.sort(key=lambda x: (x.alg, x.alg_round))
        logger.info(f'Accumulated {len(alg_acc)} alg records')
        generated_data = []

        # Generate
        for rec in alg_acc:
            js = copy.deepcopy(rec.js)
            blen = rec.tv_size

            # if alg_round is None:
            #     logger.info(f'null rounds: {alg} path: {ipath}')

            for size_mibs, ix, src_type in itertools.product(
                    [10, 100],
                    list(range(self.args.ec)),
                    ['ctr', 'hw', 'sac'],
            ):

                jse = copy.deepcopy(js)
                srnd = rec.alg_round if rec.alg_round is not None else 'x'
                jse['note'] = f'{eprefix}{rec.alg}-t:{rec.alg_type}-r:{srnd}-b:{rec.tv_size}-' \
                              f's:{size_mibs}MiB-e:{ix}-i:{src_type}'

                stream = jse['stream']
                jse['seed'] = '%016x' % ix  # normalize seed, deterministic

                # input size
                if size_mibs == 10:
                    str_sz = '10MiB'
                    data_size = SZ_10_MIB
                    jse['tv_count'] = int(math.ceil(SZ_10_MIB / blen))

                else:
                    str_sz = '100MiB'
                    data_size = SZ_100_MIB
                    jse['tv_count'] = int(math.ceil(SZ_100_MIB / blen))

                # input type
                if src_type == 'ctr':
                    jse['note'] += f'-{blen * 8}sbit-offset-{ix}'
                    inp_js = make_ctr_core(blen, '%02x' % ix)

                elif src_type == 'sac':
                    jse['note'] += f'-{blen * 8}sbit'
                    inp_js = {'type': 'sac'}

                elif src_type == 'hw':
                    weight_comp = comp_hw_weight(blen, self.args.ec, min_data=data_size)
                    try:
                        tv_count, offset, weight, offset_idx, offset_range, rem_vectors, gen_data_mb = comp_hw_data(
                            blen, weight_comp, None, jse['tv_count'], ix / self.args.ec, data_size
                        )
                    except Exception as e:
                        logger.error(f"Exception in hw gen {e}, ipath: {rec.ipath}, wcomp: {weight_comp}, tv-size: {blen}, dsize {data_size} {str_sz}", exc_info=e)
                        raise

                    inp_js = make_hw_core(offset, weight_comp)
                    src_note = '%sbit-hw%s-offsetidx-%s-offset-%s-r%.2f-vecsize-%s-%s' % (
                           blen * 8, weight, offset_idx, '-'.join(map(str, offset)), offset_range, rem_vectors, str_sz)

                    jse['note'] += f'-{src_note}'

                else:
                    raise ValueError('Unknown src_type %s' % (src_type, ))

                if rec.alg_type == 'hash':
                    stream['source'] = inp_js
                elif rec.alg_type == 'prng':
                    stream['seeder'] = inp_js
                else:
                    stream['plaintext'] = inp_js

                fname = jse['note'].replace(':', '_') + '.json'
                generated_data.append((
                    rec, jse, fname, size_mibs, ix, src_type
                ))

                if self.args.out:
                    with open(os.path.join(self.args.out, fname), 'w+') as fh:
                        json.dump(jse, fh, indent=2)
                else:
                    print(fname)

        # Generate submit experiment script
        if self.args.out:
            # all-in-one
            self.gen_submit_file(generated_data, self.args.out, '__enqueue_all.sh')
            # size-wise
            self.gen_submit_file([x for x in generated_data if x[3] == 10], self.args.out, '__enqueue_10mb.sh')
            self.gen_submit_file([x for x in generated_data if x[3] == 100], self.args.out, '__enqueue_100mb.sh')
            # experiment-wise
            for ctr in range(self.args.ec):
                self.gen_submit_file([x for x in generated_data if x[4] == ctr], self.args.out, '__enqueue_e%s.sh' % ctr)
                self.gen_submit_file([x for x in generated_data if x[4] == ctr and x[3] == 10], self.args.out, '__enqueue_e%s_10mb.sh' % ctr)
                self.gen_submit_file([x for x in generated_data if x[4] == ctr and x[3] == 100], self.args.out, '__enqueue_e%s_100mb.sh' % ctr)

    def gen_submit_file(self, collection, fold, fname):
        with open(os.path.join(fold, fname), 'w+') as fh:
            self.gen_submit_ex(collection, fh)

    def gen_submit_ex(self, collection, fh=None):
        fh = fh or io.StringIO()
        fh.write("#!/bin/bash\n")
        for rec, jse, fname, size_mibs, ix, src_type in collection:
            fh.write("submit_experiment --all_batteries "
                     "--name '%s' "
                     "--cfg '/home/debian/rtt-home/RTTWebInterface/media/predefined_configurations/%sMB.json' "
                     "--cryptostreams-config '%s'\n" % (jse['note'], size_mibs, fname))


def main():
    TaroProc().work()


if __name__ == '__main__':
    main()
