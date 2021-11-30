#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import List

import scipy.misc
from functools import lru_cache
import math
import itertools
import functools
import binascii
import json
import logging
import coloredlogs

if hasattr(scipy.misc, 'comb'):
    scipy_comb = scipy.misc.comb
else:
    import scipy.special
    scipy_comb = scipy.special.comb


logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


MODULI = {
    'F61': 2**61 + 20 * 2**32 + 1,  # 0x2000001400000001
    'F81': 2**81 + 80 * 2**64 + 1,  # 0x200500000000000000001
    'F91': 2**91 + 5 * 2**64 + 1,  # 0x80000050000000000000001
    'F125': 2**125 + 266 * 2**64 + 1,  # 0x200000000000010a0000000000000001
    'F161': 2**161 + 23 * 2**128 + 1,  # 0x20000001700000000000000000000000000000001
    'F253': 2**253 + 2**199 + 1,    # 0x2000000000000080000000000000000000000000000000000000000000000001
    'F_PBN254': 0x2523648240000001BA344D80000000086121000000000013A700000000000013,
    'F_QBN254': 0x2523648240000001BA344D8000000007FF9F800000000010A10000000000000D,
    'F_QBN128': 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001,
    'F_PBLS12_381': 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab,
    'F_QBLS12_381': 0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001,  # 255 bits
    'F_PED25519': 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed,
    'F_QED25519': 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed,
    'Bin63': 2**63,
    'Bin81': 2**81,
    'Bin91': 2**91,
    'Bin127': 2**127,
    'Bin161': 2**161,
    'Bin255': 2**255,
}

# Input lengths for PRNGs
ILENS = {
    'std_mersenne_twister': 4,
    'std_lcg': 4,
    'std_subtract_with_carry': 4,
    'testu01-ulcg': 6,  # estimate, precise number is computed from constants
    'testu01-umrg': 6,  # estimate, precise number is computed from constants
    'testu01-uxorshift': 8 * 4,
}


class LowmcParams:
    def __init__(self, name, key_size, block_size, sboxes, full_rounds):
        self.name = name
        self.key_size = key_size
        self.block_size = block_size
        self.sboxes = sboxes
        self.full_rounds = full_rounds


LOWMC_PARAMS = {
    'lowmc-s80a': LowmcParams('lowmc-s80a', 80, 256, 49, full_rounds=12),
    'lowmc-s80b': LowmcParams('lowmc-s80b', 80, 128, 31, full_rounds=12),
    'lowmc-s128a': LowmcParams('lowmc-s128a', 128, 256, 63, full_rounds=14),
    'lowmc-s128b': LowmcParams('lowmc-s128b', 128, 128, 1, full_rounds=252),
    'lowmc-s128c': LowmcParams('lowmc-s128c', 128, 128, 2, full_rounds=128),
    'lowmc-s128d': LowmcParams('lowmc-s128d', 128, 128, 3, full_rounds=88),
}  # type: dict[str, LowmcParams]


class MpcSageParams:
    def __init__(self, name, field, full_rounds, script_name, round_tpl='-r %s'):
        self.name = name
        self.field = field
        self.full_rounds = full_rounds
        self.script = script_name
        self.round_tpl = round_tpl


_ROUND_TPL_PS = '--rf 2 --rp 0 --red-rf1 %s --red-rf2 %s --red-rp %s'
MPC_SAGE_PARAMS = {
    'Poseidon_S80b': MpcSageParams('Poseidon_S80b', 'F161', (8, 50), 'starkad_poseidon.sage', _ROUND_TPL_PS),
    'Poseidon_S128a': MpcSageParams('Poseidon_S128a', 'F125', (8, 81), 'starkad_poseidon.sage', _ROUND_TPL_PS),
    'Poseidon_S128b': MpcSageParams('Poseidon_S128b', 'F253', (8, 83), 'starkad_poseidon.sage', _ROUND_TPL_PS),
    'Poseidon_S128c': MpcSageParams('Poseidon_S128c', 'F125', (8, 83), 'starkad_poseidon.sage', _ROUND_TPL_PS),
    'Poseidon_S128d': MpcSageParams('Poseidon_S128d', 'F61', (8, 40), 'starkad_poseidon.sage', _ROUND_TPL_PS),
    'Poseidon_S128e': MpcSageParams('Poseidon_S128e', 'F253', (8, 85), 'starkad_poseidon.sage', _ROUND_TPL_PS),
    'Poseidon_S128_BLS12_138': MpcSageParams('Poseidon_S128_BLS12_138', 'F_QBLS12_381', (8, 60), 'starkad_poseidon.sage', _ROUND_TPL_PS),

    'Starkad_S80b': MpcSageParams('Starkad_S80b', 'Bin161', (8, 52), 'starkad_poseidon.sage', _ROUND_TPL_PS),
    'Starkad_S128a': MpcSageParams('Starkad_S128a', 'Bin127', (8, 85), 'starkad_poseidon.sage', _ROUND_TPL_PS),
    'Starkad_S128b': MpcSageParams('Starkad_S128b', 'Bin255', (8, 88), 'starkad_poseidon.sage', _ROUND_TPL_PS),
    'Starkad_S128c': MpcSageParams('Starkad_S128c', 'Bin127', (8, 86), 'starkad_poseidon.sage', _ROUND_TPL_PS),
    'Starkad_S128d': MpcSageParams('Starkad_S128d', 'Bin63', (8, 43), 'starkad_poseidon.sage', _ROUND_TPL_PS),
    'Starkad_S128e': MpcSageParams('Starkad_S128e', 'Bin255', (8, 88), 'starkad_poseidon.sage', _ROUND_TPL_PS),

    'Rescue_S45a': MpcSageParams('Rescue_S45a', 'F91', (10,), 'vision.sage'),
    'Rescue_S45b': MpcSageParams('Rescue_S45b', 'F91', (10,), 'vision.sage'),
    'Rescue_S80a': MpcSageParams('Rescue_S80a', 'F81', (10,), 'vision.sage'),
    'Rescue_S80b': MpcSageParams('Rescue_S80b', 'F161', (10,), 'vision.sage'),
    'Rescue_S128a': MpcSageParams('Rescue_S128a', 'F125', (16,), 'vision.sage'),
    'Rescue_S128b': MpcSageParams('Rescue_S128b', 'F253', (22,), 'vision.sage'),
    'Rescue_S128e': MpcSageParams('Rescue_S128e', 'F253', (10,), 'vision.sage'),

    'Vision_S45a': MpcSageParams('Vision_S45a', 'Bin91', (10,), 'vision.sage'),
    'Vision_S45b': MpcSageParams('Vision_S45b', 'Bin91', (10,), 'vision.sage'),
    'Vision_S80a': MpcSageParams('Vision_S80a', 'Bin81', (10,), 'vision.sage'),
    'Vision_S80b': MpcSageParams('Vision_S80b', 'Bin161', (10,), 'vision.sage'),
    'Vision_S128a': MpcSageParams('Vision_S128a', 'Bin127', (12,), 'vision.sage'),
    'Vision_S128b': MpcSageParams('Vision_S128b', 'Bin255', (26,), 'vision.sage'),
    'Vision_S128d': MpcSageParams('Vision_S128d', 'Bin63', (10,), 'vision.sage'),

    'S45a': MpcSageParams('S45a', 'F91', (121,), 'gmimc.sage'),
    'S45b': MpcSageParams('S45b', 'F91', (137,), 'gmimc.sage'),
    'S80a': MpcSageParams('S80a', 'F81', (111,), 'gmimc.sage'),
    'S80b': MpcSageParams('S80b', 'F161', (210,), 'gmimc.sage'),
    'S128a': MpcSageParams('S128a', 'F125', (166,), 'gmimc.sage'),
    'S128e': MpcSageParams('S128e', 'F253', (342,), 'gmimc.sage'),

    'S45': MpcSageParams('S45', 'F91', (116,), 'mimc_hash.sage'),
    'S80': MpcSageParams('S80', 'F161', (204,), 'mimc_hash.sage'),
    'S128': MpcSageParams('S128', 'F253', (320,), 'mimc_hash.sage'),
}  # type: dict[str, MpcSageParams]


class ExpRec:
    def __init__(self, ename, ssize, fname, tpl_file, cfg_type=None):
        self.ename = ename
        self.ssize = ssize
        self.fname = fname
        self.tpl_file = tpl_file
        self.cfg_type = cfg_type

    def __eq__(self, o: object) -> bool:
        return (self.ename, self.ssize, self.fname, self.cfg_type, self.tpl_file) \
               == (o.ename, o.ssize, o.fname, o.cfg_type, o.tpl_file) if isinstance(o, ExpRec) else super().__eq__(o)

    def __repr__(self) -> str:
        return f'ExpRec({self.ename}, {self.ssize}, {self.fname}, {self.cfg_type})'

    def __hash__(self) -> int:
        return hash((self.ename, self.ssize, self.fname, self.cfg_type, self.tpl_file))


class StreamRec:
    def __init__(self, stype=None, sdesc=None, sscript=None, expid=None, seed=None):
        self.stype = stype
        self.sdesc = sdesc
        self.sscript = sscript
        self.expid = expid
        self.seed = seed


class HwConfig:
    def __init__(self, script=None, core=None, weight=None, offset=None,
                 offset_range=None, rem_vectors=None, gen_data_mb=None, note=None):
        self.script = script
        self.core = core
        self.weight = weight
        self.offset = offset
        self.offset_range = offset_range
        self.rem_vectors = rem_vectors
        self.gen_data_mb = gen_data_mb
        self.note = note


class StreamOptions:
    ZERO = 1
    CTR = 2
    LHW = 4
    SAC = 8
    RND = 16

    CTR_LHW = CTR | LHW
    CTR_LHW_SAC = CTR_LHW | SAC
    CTR_LHW_SAC_RND = CTR_LHW_SAC | RND
    SAC_RND = SAC | RND

    @staticmethod
    def has_zero(x):
        return (x & StreamOptions.ZERO) > 0

    @staticmethod
    def has_ctr(x):
        return (x & StreamOptions.CTR) > 0

    @staticmethod
    def has_lhw(x):
        return (x & StreamOptions.LHW) > 0

    @staticmethod
    def has_sac(x):
        return (x & StreamOptions.SAC) > 0

    @staticmethod
    def has_rnd(x):
        return (x & StreamOptions.RND) > 0


@lru_cache(maxsize=1024)
def comb(n, k, exact=False):
    return scipy_comb(n, k, exact=exact)


@lru_cache(maxsize=8192)
def comb_cached(n, k):
    """
    Computes C(n,k) - combinatorial number of N elements, k choices
    :param n:
    :param k:
    :return:
    """
    if (k > n) or (n < 0) or (k < 0):
        return 0
    val = 1
    for j in range(min(k, n - k)):
        val = (val * (n - j)) // (j + 1)
    return val


def rank(s, n):
    """
    Returns index of the combination s in (N,K)
    https://computationalcombinatorics.wordpress.com/2012/09/10/ranking-and-unranking-of-combinations-and-permutations/
    :param s:
    :return:
    """
    k = len(s)
    r = 0
    for i in range(0, k):
        for v in range(s[i-1]+1 if i > 0 else 0, s[i]):
            r += comb_cached(n - v - 1, k - i - 1)
    return r


@lru_cache(maxsize=8192)
def unrank(i, n, k):
    """
    returns the i-th combination of k numbers chosen from 0,2,...,n-1, indexing from 0
    """
    c = []
    r = i+0
    j = 0
    for s in range(1, k+1):
        cs = j+1

        while True:
            if n - cs < 0:
                raise ValueError('Invalid index')
            decr = comb(n - cs, k - s)
            if r > 0 and decr == 0:
                raise ValueError('Invalid index')
            if r - decr >= 0:
                r -= decr
                cs += 1
            else:
                break

        c.append(cs-1)
        j = cs
    return c


def get_input_key(alg_type):
    """Returns cstreams algorithm input js config key"""
    if alg_type == 'hash':
        return 'source'
    elif alg_type == 'prng':
        return 'seeder'  # for PRNGs we feed seeder / key
    else:
        return 'plaintext'


def get_input_size(config):
    """Returns input block size for the cstreams config"""
    alg_type = config['stream']['type']
    if alg_type == 'hash':
        return config['stream']['input_size']
    elif alg_type == 'prng':
        return ILENS[config['stream']['algorithm']]
    else:
        return config['stream']['block_size']


def make_ctr_config(blen=31, offset='00', tv_count=None, min_data=None, core_only=False) -> dict:
    """
    Generate counter CryptoStreams config with configurable offset.
     - blen is block width in bytes
     - offset is MSB hex-coded byte to distinguish counter sequences
     - tv_count is desired number of output blocks. If None, set to the maximal value
     - min_data if set, require that number of data generated by the script is at least min_data, raises otherwise
    """
    max_vals = 2**((blen-1)*8) - 1
    tv_count = min(tv_count or max_vals, 2**62)
    data_mb = blen * tv_count / 1024 / 1024

    if len(offset) != 2:
        raise ValueError('Offset has to be hex-coded byte')

    if max_vals < tv_count:
        logger.info('TV count is higher than ctr space %s bits, max vals: %s'
                    % (blen-1, max_vals))

    if min_data is not None and min_data > blen * min(max_vals, tv_count):
        raise ValueError('Condition on minimal data could not be fulfilled')

    note = 'plaintext-ctr-%sbit-%.2fMB' % (blen*8, data_mb)
    if core_only:
        return make_ctr_core(blen, offset)

    ctr_file = {
        "notes": note,
        "seed": "0000000000000000",
        "tv-size": None,
        "tv-count": None,
        "tv_size": blen,
        "tv_count": tv_count,
        "stdout": True,
        "file_name": "plain_ctr.bin",
        "stream": make_ctr_core(blen, offset)
      }

    return ctr_file


def make_ctr_core(blen, offset='00'):
    return {
        "type": "xor_stream",
        "note": "ctr-offset",
        "ctr_offset": int(offset, 16),
        "source": {
            "type": "tuple_stream",
            "sources": [
                {
                    "type": "counter",
                    "output_size": blen
                },
                {
                    "type": "single_value_stream",
                    "output_size": blen,
                    "source": {
                        "type": "tuple_stream",
                        "sources": [
                            {
                                "type": "false_stream",
                                "output_size": blen - 1
                            },
                            {
                                "type": "const_stream",
                                "output_size": 1,
                                "value": offset
                            }
                        ]
                    }
                }
            ]
        }
    }


def make_hw_config(blen=31, weight=4, offset=None, tv_count=None, offset_range: float = None, min_data=None,
                   core_only=False, return_aux=False):
    """
    Generate HW counter CryptoStreams config with configurable offset.
     - blen is block width in bytes
     - weight is number of bits enabled
     - offset is initial combination array, |offset| == weight
     - tv_count is desired number of output blocks. If None, set to the maximal value
     - offset_range is float in [0, 1), computes offset to start in the given combination index
     - min_data if set, require that number of data generated by the script is at least min_data, raises otherwise
    """
    if offset is not None and weight != len(offset):
        raise ValueError('Offset length error')

    offset = offset or list(range(weight))
    num_vectors = comb_cached(blen*8, weight)

    if offset_range is not None:
        offset_idx = int(num_vectors * offset_range)
        offset = unrank(offset_idx, blen*8, weight)
        logger.debug('Offset computed from range %s, index %s, offset: %s'
                     % (offset_range, offset_idx, offset))
    else:
        offset_idx = rank(offset, blen*8)
        offset_range = offset_idx / num_vectors

    rem_vectors = num_vectors - offset_idx
    tv_count = min(tv_count or rem_vectors, 2**62)
    gen_data_mb = tv_count * blen / 1024 / 1024

    if tv_count > rem_vectors:
        logger.info('Number of generatable vectors is lower than desired. '
                    'Num: %s, offset: %s, remains: %s. Offset: %s. Max generated data: %.2f MB'
                    % (num_vectors, offset_idx, rem_vectors, offset, gen_data_mb))

    if min_data is not None and min_data > blen * min(rem_vectors, tv_count) * blen:
        raise ValueError('Condition on minimal data could not be fulfilled')

    note = 'hw-%sbit-hw%s-offsetidx-%s-offset-%s-r%.2f-vecsize-%s-%.2fMB' \
           % (blen * 8, weight, offset_idx, '-'.join(map(str, offset)), offset_range, rem_vectors, gen_data_mb)

    core = {
      "type": "hw_counter",
      "initial_state": offset,
      "hw": weight,
      "offset_ratio": offset_range,
    }

    hw_file = {
        "notes": 'plaintext-' + note,
        "seed": "0000000000000000",
        "tv-size": None,
        "tv-count": None,
        "tv_size": blen,
        "tv_count": tv_count,
        "stdout": True,
        "file_name": "hw_ctr.bin",
        "stream": core
    }

    if return_aux:
        return HwConfig(script=hw_file, core=core, weight=weight, offset=offset, offset_range=offset_range,
                        rem_vectors=rem_vectors, gen_data_mb=gen_data_mb, note=note)

    cfg_to_return = core if core_only else hw_file
    return cfg_to_return


def make_hw_core(offset, weight):
    return {
        "type": "hw_counter",
        "initial_state": offset,
        "hw": weight
    }


def get_strategy_prime(modulus, ob, ib=256, st=6, max_out=None, seed=None):
    """
    Generate spreader strategy command for prime fields
      - max_out is number of output bits
    """
    r = '-m %s --ob=%s --ib=%s --stdin --rgen aes -s=%s --st %s --max-out %s' \
        % (hex(modulus), ob, ib, seed or 'ff55', st, max_out or 88080384)
    return r


def get_strategy_binary(ob, fob=None, ib=256, st=6, max_out=None, seed=None):
    """
    Generate spreader strategy command for binary fields
    """
    md = 2**(fob or ob)
    r = '-m %s --ob=%s --ib=%s --stdin --rgen aes -s=%s --st %s --max-out %s' \
        % (hex(md), ob, ib, seed or 'ff55', st, max_out or 88080384)
    return r


def get_strategy_binary_raw(ob, fob=None, ib=None, max_out=None):
    """Raw strategy, just truncate to max_out bits"""
    return get_strategy_binary(ob=ob, fob=fob, ib=ib or ob, st=0, max_out=max_out)


def get_strategy_binary_expand(ob, fob, ib=None, max_out=None):
    """Expands, add bit. fob = function output bits, e.g., 255, ob = 256 -> add 1 msb"""
    return get_strategy_prime(modulus=2**fob, ob=ob, ib=ib, max_out=max_out, st=18)


def get_strategy_binary_trunc(ob, fob=None, ib=None, max_out=None):
    """Truncates"""
    return get_strategy_binary(ob=ob, fob=fob, ib=ib, max_out=max_out, st=13)


def get_strategy_prime_expdrop6(modulus, ob, ib=None, max_out=None):
    """Expands to osize, mod < osize"""
    if modulus > 2**ob:
        logger.warning('Modulus is greater than ob range ob: %s, mod bits: %s, mod: %s'
                       % (ob, math.log(modulus, 2), hex(modulus)))
    return get_strategy_prime(modulus=modulus, ob=ob, ib=ib, st=6, max_out=max_out)


def get_strategy_prime_expinv15(modulus, ob, ib=None, max_out=None):
    """Expands to osize, mod < osize"""
    if modulus > 2**ob:
        logger.warning('Modulus is greater than ob range ob: %s, mod bits: %s, mod: %s'
                       % (ob, math.log(modulus, 2), hex(modulus)))
    return get_strategy_prime(modulus=modulus, ob=ob, ib=ib, st=15, max_out=max_out)


def get_strategy_prime_drop16(modulus, ob, ib=None, max_out=None):
    """Accepts only output < 2**ob, modulus has to be bigger"""
    if modulus < 2**ob:
        logger.warning('Modulus is smaller than ob range')
    return get_strategy_prime(modulus=modulus, ob=ob, ib=ib, st=16, max_out=max_out)


def get_smaller_byteblock(bits):
    return bits // 8


class HwSpaceTooSmall(ValueError):
    def __init__(self, *args, **kwargs):
        super(HwSpaceTooSmall, self).__init__(*args, **kwargs)


def comp_hw_weight(blen, samples=3, min_data=None, min_samples=None):
    for hw in range(3, blen*8 // 2):
        max_combs = comb_cached(blen*8, hw)
        cur_comb = max_combs / samples

        if min_data is not None and cur_comb * blen < min_data:
            continue

        if min_samples is not None and cur_comb < min_samples:
            continue

        return hw

    raise HwSpaceTooSmall('Could not find suitable HW')


def log2ceil(x):
    cd = math.ceil(math.log(x, 2))
    return cd if x <= 2**cd else cd+1


def comp_rejection_ratio_st6(modulus, osize_interval):
    """Returns a probability that a random draw from [0, modulus) will be rejected when spreading
    to uniform distribution to [0, osize_interval). Holds only for modulus <= osize_interval"""
    tp = math.ceil(osize_interval / modulus) - 1  # number of full-sized ms inside the range
    return 1 / (tp + 1) * ((tp + 1) * modulus - osize_interval) / modulus


def augment_round_configs(to_gen):
    res = []
    for x in to_gen:
        for r in x[-1]:
            res.append(tuple(list(x[:-1]) + [r]))
    return res


def gen_posseidon(data_sizes=None, eprefix=None, streams=StreamOptions.CTR_LHW):
    rstsr = '--rf 2 --rp 0 --red-rf1 %s --red-rf2 %s --red-rp %s'

    # fname, field name, rounds structure, sage file, round string, rounds to test
    to_gen = [
        ('Poseidon_S80b', 'F161', (8, 50), 'starkad_poseidon.sage', rstsr, [(1, 0, 0)]),
        # ('Poseidon_S128a', 'F125', (8, 81), 'starkad_poseidon.sage', rstsr, [(1, 0, 0)]),
        # ('Poseidon_S128b', 'F253', (8, 83), 'starkad_poseidon.sage', rstsr, [(1, 0, 0)]),
        # ('Poseidon_S128c', 'F125', (8, 83), 'starkad_poseidon.sage', rstsr, [(1, 0, 0)]),
        # ('Poseidon_S128d', 'F61', (8, 40), 'starkad_poseidon.sage', rstsr, [(1, 0, 0)]),
        # ('Poseidon_S128e', 'F253', (8, 85), 'starkad_poseidon.sage', rstsr, [(1, 0, 0)]),
        ('Poseidon_S128_BLS12_138', 'F_QBLS12_381', (8, 60), 'starkad_poseidon.sage', rstsr, [(1, 0, 0)]),
    ]
    return gen_prime_config(to_gen, data_sizes, eprefix=eprefix, streams=streams)


def gen_starkad(data_sizes=None, eprefix=None, to_gen=None, streams=StreamOptions.CTR_LHW):
    rstsr = '--rf 2 --rp 0 --red-rf1 %s --red-rf2 %s --red-rp %s'

    # fname, field name, rounds structure, sage file, round string, rounds to test
    to_gen = to_gen if to_gen else [
        ('Starkad_S80b', 'Bin161', (8, 52), 'starkad_poseidon.sage', rstsr, [(1, 0, 0)]),
        # ('Starkad_S128a', 'Bin127', (8, 85), 'starkad_poseidon.sage', rstsr, [(1, 0, 0)]),
        # ('Starkad_S128b', 'Bin255', (8, 88), 'starkad_poseidon.sage', rstsr, [(1, 0, 0)]),
        # ('Starkad_S128c', 'Bin127', (8, 86), 'starkad_poseidon.sage', rstsr, [(1, 0, 0)]),
        # ('Starkad_S128d', 'Bin63', (8, 43), 'starkad_poseidon.sage', rstsr, [(1, 0, 0)]),
        ('Starkad_S128e', 'Bin255', (8, 88), 'starkad_poseidon.sage', rstsr, [(1, 0, 0)]),
    ]
    return gen_binary_config(to_gen, data_sizes, eprefix=eprefix, streams=streams)


def gen_rescue(data_sizes=None, eprefix=None, to_gen=None, streams=StreamOptions.CTR_LHW):
    rstsr = '-r %s'

    # fname, field name, rounds structure, sage file, round string, rounds to test
    to_gen = to_gen if to_gen else [
        ('Rescue_S45a', 'F91', (10,), 'vision.sage', rstsr, [(1,), (2,)]),
        # ('Rescue_S45b', 'F91', (10,), 'vision.sage', rstsr, [(1,), (2,)]),
        # ('Rescue_S80a', 'F81', (10,), 'vision.sage', rstsr, [(1,), (2,)]),
        # ('Rescue_S80b', 'F161', (10,), 'vision.sage', rstsr, [(1,), (2,)]),
        # ('Rescue_S128a', 'F125', (16,), 'vision.sage', rstsr, [(1,), (2,)]),
        # ('Rescue_S128b', 'F253', (22,), 'vision.sage', rstsr, [(1,), (2,)]),
        ('Rescue_S128e', 'F253', (10,), 'vision.sage', rstsr, [(1,), (2,)]),
    ]
    return gen_prime_config(to_gen, data_sizes, eprefix=eprefix, streams=streams)


def gen_vision(data_sizes=None, eprefix=None, to_gen=None, streams=StreamOptions.CTR_LHW):
    rstsr = '-r %s'

    # fname, field name, rounds structure, sage file, round string, rounds to test
    to_gen = to_gen if to_gen else [
        ('Vision_S45a', 'Bin91', (10,), 'vision.sage', rstsr, [(1,), (2,)]),
        # ('Vision_S45b', 'Bin91', (10,), 'vision.sage', rstsr, [(1,), (2,)]),
        # ('Vision_S80a', 'Bin81', (10,), 'vision.sage', rstsr, [(1,), (2,)]),
        # ('Vision_S80b', 'Bin161', (10,), 'vision.sage', rstsr, [(1,), (2,)]),
        # ('Vision_S128a', 'Bin127', (12,), 'vision.sage', rstsr, [(1,), (2,)]),
        # ('Vision_S128b', 'Bin255', (26,), 'vision.sage', rstsr, [(1,), (2,)]),
        ('Vision_S128d', 'Bin63', (10,), 'vision.sage', rstsr, [(1,), (2,)]),
    ]
    return gen_binary_config(to_gen, data_sizes, eprefix=eprefix, streams=streams)


def gen_gmimc(data_sizes=None, eprefix=None, to_gen=None, streams=StreamOptions.CTR_LHW):
    rstsr = '-r %s'

    # fname, field name, rounds structure, sage file, round string, rounds to test
    to_gen = to_gen if to_gen else [
        ('S45a', 'F91', (121,), 'gmimc.sage', rstsr, [(1,), (2,)]),
        # ('S45b', 'F91', (137,), 'gmimc.sage', rstsr, [(1,), (2,)]),
        # ('S80a', 'F81', (111,), 'gmimc.sage', rstsr, [(1,), (2,)]),
        # ('S80b', 'F161', (210,), 'gmimc.sage', rstsr, [(1,), (2,)]),
        # ('S128a', 'F125', (166,), 'gmimc.sage', rstsr, [(1,), (2,)]),
        ('S128e', 'F253', (342,), 'gmimc.sage', rstsr, [(1,), (2,)]),
    ]
    return gen_prime_config(to_gen, data_sizes, eprefix=eprefix, streams=streams)


def gen_mimc(data_sizes=None, eprefix=None, to_gen=None, streams=StreamOptions.CTR_LHW):
    rstsr = '-r %s'

    # fname, field name, rounds structure, sage file, round string, rounds to test
    to_gen = to_gen if to_gen else [
        ('S45', 'F91', (116,), 'mimc_hash.sage', rstsr, [(1,), (2,)]),
        ('S80', 'F161', (204,), 'mimc_hash.sage', rstsr, [(1,), (2,)]),
        ('S128', 'F253', (320,), 'mimc_hash.sage', rstsr, [(1,), (2,)]),
    ]
    return gen_prime_config(to_gen, data_sizes, eprefix=eprefix, streams=streams)


def gen_all(data_sizes=None, eprefix=None):
    return gen_posseidon(data_sizes, eprefix) \
           + gen_starkad(data_sizes, eprefix) \
           + gen_rescue(data_sizes, eprefix) \
           + gen_vision(data_sizes, eprefix) \
           + gen_gmimc(data_sizes, eprefix) \
           + gen_mimc(data_sizes, eprefix) \
           + gen_lowmc(data_sizes, eprefix)


def get_prime_strategies(moduli, moduli_bits, out_block_bits, max_out_b):
    return [
        # fit to moduli size; 6: rand offset window, rejection sample on moduli.
        # ob = moduli_bits, spread to 2**moduli_bits, fill whole modulus, does not have to be byte-aligned
        ('s6mb', get_strategy_prime_expdrop6(moduli, ob=moduli_bits, ib=out_block_bits, max_out=max_out_b)),

        # fit to moduli size, byte align top
        # ob = out_block_bits, moduli bits, byte aligned, contains whole modulus
        ('s6ob', get_strategy_prime_expdrop6(moduli, ob=out_block_bits, ib=out_block_bits, max_out=max_out_b)),

        # fit to moduli size; 15: spread_inverse_frac, inverse sampling with arbitrary precision
        # 15: stretch to whole interval by scaling, fill gaps from stretch by uniform number dist, arbitrary prec.
        ('s15mb', get_strategy_prime_expinv15(moduli, ob=moduli_bits, ib=out_block_bits, max_out=max_out_b)),

        # fit to moduli size, byte align top
        ('s15ob', get_strategy_prime_expinv15(moduli, ob=out_block_bits, ib=out_block_bits, max_out=max_out_b)),

        # one bit smaller than moduli, 2x more data
        # 16: simple rejection sampling on 2**(moduli_bits - 1), cuts top bit
        ('s16mb1', get_strategy_prime_drop16(moduli, ob=moduli_bits - 1, ib=out_block_bits, max_out=max_out_b))
    ]


def get_prime_accepting_ratio(moduli, moduli_bits, out_block_bits):
    return 1 - max(
        comp_rejection_ratio_st6(moduli, 2 ** moduli_bits),
        comp_rejection_ratio_st6(moduli, 2 ** out_block_bits),
        0.5  # st16, half width reduction
    )


def get_binary_strategies(moduli_bits, out_block_bits, max_out_b):
    bnds = [64, 128, 324, 256, 512, 1024, 2048]
    nb = [x for x in bnds if x > out_block_bits][0]

    return [
        # raw
        ('s0mb', get_strategy_binary_raw(fob=moduli_bits, ob=moduli_bits, ib=out_block_bits, max_out=max_out_b)),

        # fit to moduli size, byte align top
        ('s18ob', get_strategy_binary_expand(fob=moduli_bits, ob=out_block_bits, ib=out_block_bits, max_out=max_out_b)),

        # fit to moduli size, byte align top
        ('s18ab', get_strategy_binary_expand(fob=moduli_bits, ob=nb, ib=out_block_bits, max_out=max_out_b)),
    ]


def get_binary_accepting_ratio():
    return 1


def gen_prime_config(to_gen, data_sizes=None, eprefix=None, streams=StreamOptions.CTR_LHW):
    return gen_script_config(to_gen, True, data_sizes=data_sizes, eprefix=eprefix, streams=streams)


def gen_binary_config(to_gen, data_sizes=None, eprefix=None, streams=StreamOptions.CTR_LHW):
    return gen_script_config(to_gen, False, data_sizes=data_sizes, eprefix=eprefix, streams=streams)


def myformat(_fmtstr, **kwargs):
    for k in kwargs:
        _fmtstr = _fmtstr.replace('{{' + k + '}}', str(kwargs[k]))
    return _fmtstr


def gen_script_config(to_gen, is_prime=True, data_sizes=None, eprefix=None, streams=StreamOptions.CTR_LHW):
    data_sizes = data_sizes or [100 * 1024 * 1024]

    tpl = '{{SAGE_BIN}} {{RTT_EXEC}}/rtt-mpc/rtt_mpc/{{sfile}} ' \
          '-f {{name}} ' \
          '--inp-block-size {{inp_block_bits}} ' \
          '--inp-block-count 1 ' \
          '--out-block-size {{out_block_bits}} ' \
          '--out-blocks 1 ' \
          '--raw ' \
          '{{rounds}} '

    full_tpl = 'cd {{RTT_EXEC}}/rtt-mpc/rtt_mpc; ' \
               '{{CRYPTOSTREAMS_BIN}} -c={{FILE_CONFIG1.JSON}} | ' \
               '{{tpl}} | ' \
               '{{PYTHON_BIN}} -m rtt_data_gen.spreader {{spreader}}'

    agg_configs = []
    agg_scripts = []
    for ccfg in itertools.product(augment_round_configs(to_gen), data_sizes):
        cpos = ccfg[0]
        max_out = ccfg[1]
        sfile = cpos[3]
        rrounds = cpos[2]  # full rounds specs

        moduli = MODULI[cpos[1]]
        moduli_bits = log2ceil(moduli)
        inp_block_bytes = get_smaller_byteblock(moduli_bits)  # byte-aligned fits all in moduli (generator, input)
        inp_block_bits = 8 * inp_block_bytes
        out_block_bits = 8 * math.ceil(moduli_bits / 8)  # byte-aligned wrapper for moduli

        ctpl = myformat(tpl, sfile=sfile, name=cpos[0],
                        inp_block_bits=inp_block_bits,
                        out_block_bits=out_block_bits,
                        rounds=cpos[4] % cpos[5])

        max_out_b = max_out * 8
        accept_ratio = get_prime_accepting_ratio(moduli, moduli_bits, out_block_bits) if is_prime \
            else get_binary_accepting_ratio()

        req_data = max_out * (1 / accept_ratio)  # data required to generate, taking rejections into account
        min_data = (req_data / ((moduli_bits - 1) // 8)) * inp_block_bytes  # compute for input widths
        if min_data < max_out or req_data < max_out:
            raise ValueError('Assertion error on min data')

        ctr_configs = [
            ('ctr00-b%s' % inp_block_bytes, make_ctr_config(inp_block_bytes, offset='00', min_data=min_data)),
            ('ctr01-b%s' % inp_block_bytes, make_ctr_config(inp_block_bytes, offset='01', min_data=min_data)),
            ('ctr02-b%s' % inp_block_bytes, make_ctr_config(inp_block_bytes, offset='02', min_data=min_data)),
        ] if StreamOptions.has_ctr(streams) else []

        weight = comp_hw_weight(inp_block_bytes, samples=3, min_data=min_data)
        hw_configs = [
            ('lhw00-b%s-w%s' % (inp_block_bytes, weight), make_hw_config(inp_block_bytes, weight=weight, offset_range=0.0, min_data=min_data)),
            ('lhw01-b%s-w%s' % (inp_block_bytes, weight), make_hw_config(inp_block_bytes, weight=weight, offset_range=1/3., min_data=min_data)),
            ('lhw02-b%s-w%s' % (inp_block_bytes, weight), make_hw_config(inp_block_bytes, weight=weight, offset_range=2/3., min_data=min_data)),
        ] if StreamOptions.has_lhw(streams) else []

        agg_inputs = ctr_configs + hw_configs
        agg_spreads = get_prime_strategies(moduli, moduli_bits, out_block_bits, max_out_b) if is_prime \
            else get_binary_strategies(moduli_bits, out_block_bits, max_out_b)

        for configs in itertools.product(agg_spreads, agg_inputs):
            inp_name = configs[1][0]
            spread_name = configs[0][0]

            inp = configs[1][1]
            cfull_tpl = myformat(full_tpl,
                                 tpl=ctpl,
                                 spreader=configs[0][1]
            )

            agg_configs.append((ctpl, inp, cfull_tpl))
            ename = '%s%s-%s-raw-r%s-inp-%s-spr-%s-s%sMB' \
                    % (eprefix or '', cpos[0], 'pri' if is_prime else 'bin',
                       '-'.join(map(str, cpos[5])), inp_name, spread_name, int(max_out/1024/1024))

            script = {
              "stream": {
                "type": "shell",
                "direct_file": False,
                "exec": cfull_tpl,
              },
              "input_files": {
                "CONFIG1.JSON": {
                  "note": inp['notes'],
                  "data": inp,
                }
              }
            }
            agg_scripts.append(ExpRec(ename=ename, ssize=max_out / 1024 / 1024, fname=ename + '.json',
                                      tpl_file=script, cfg_type='rtt-data-gen-config'))
    return agg_scripts


def gen_lowmc(data_sizes=None, eprefix=None, streams=StreamOptions.CTR_LHW):
    """lowmc operates over binary field / whole bit blocks"""

    to_gen = [
        # name, rounds-all, rounds to try
        ('lowmc-s80a', (12, ), [3, 4, 5, 6, 7, 8, 9, 10, 11]),
        ('lowmc-s80b', (12, ), [3, 4, 5, 6, 7, 8, 9, 10, 11]),
        ('lowmc-s128a', (14, ), [3, 4, 5, 6, 7, 8, 9, 10, 11]),
        ('lowmc-s128b', (252, ), [3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]),
        ('lowmc-s128c', (128, ), [3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]),
        ('lowmc-s128d', (88, ), [3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 30, 40, 50, 60, 70, 80]),

        # ('lowmc-s128b', (252, ), [146, 151, 152, 153, 154, 155, 156, 157, 158, 159]),
        # ('lowmc-s128c', (128, ), [108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]),
        # ('lowmc-s128d', (88, ), [59, 61, 62, 63, 64, 65]),
    ]

    return gen_lowmc_core(to_gen, data_sizes, eprefix, streams)


def gen_lowmc_core(to_gen, data_sizes=None, eprefix=None, streams=StreamOptions.CTR_LHW):
    data_sizes = data_sizes or [100 * 1024 * 1024]
    full_tpl = '{{CRYPTOSTREAMS_BIN}} -c={{FILE_CONFIG1.JSON}}'

    agg_configs = []
    agg_scripts = []
    for ccfg in itertools.product(augment_round_configs(to_gen), data_sizes):
        cpos = ccfg[0]
        cname = cpos[0]
        max_out = ccfg[1]
        cround = cpos[-1]

        inp_block_bytes = LOWMC_PARAMS[cname].block_size // 8
        key_size = LOWMC_PARAMS[cname].key_size // 8
        sboxes = LOWMC_PARAMS[cname].sboxes
        min_data = max_out
        tv_count = int(math.ceil(8*max_out / LOWMC_PARAMS[cname].block_size))

        ctr_configs = [
            ('ctr00-b%s' % inp_block_bytes, make_ctr_config(inp_block_bytes, offset='00', min_data=min_data), '0000000000000000'),
            ('ctr01-b%s' % inp_block_bytes, make_ctr_config(inp_block_bytes, offset='01', min_data=min_data), '0000000000000001'),
            ('ctr02-b%s' % inp_block_bytes, make_ctr_config(inp_block_bytes, offset='02', min_data=min_data), '0000000000000002'),
        ] if StreamOptions.has_ctr(streams) else []

        weight = comp_hw_weight(inp_block_bytes, samples=3, min_data=min_data)
        hw_configs = [
            ('lhw00-b%s-w%s' % (inp_block_bytes, weight),
             make_hw_config(inp_block_bytes, weight=weight, offset_range=0.0, min_data=min_data), '0000000000000003'),
            ('lhw01-b%s-w%s' % (inp_block_bytes, weight),
             make_hw_config(inp_block_bytes, weight=weight, offset_range=1/3., min_data=min_data), '0000000000000004'),
            ('lhw02-b%s-w%s' % (inp_block_bytes, weight),
             make_hw_config(inp_block_bytes, weight=weight, offset_range=2/3., min_data=min_data), '0000000000000005'),
        ] if StreamOptions.has_lhw(streams) else []

        sac_configs = [
            ('sac00-b%s' % inp_block_bytes, {'type': 'sac'}, '0000000000000006'),
            ('sac01-b%s' % inp_block_bytes, {'type': 'sac'}, '0000000000000007'),
            ('sac02-b%s' % inp_block_bytes, {'type': 'sac'}, '0000000000000008'),
        ] if StreamOptions.has_sac(streams) else []

        agg_inputs = ctr_configs + hw_configs + sac_configs
        agg_spreads = [('', None)]  # get_binary_strategies(moduli_bits, out_block_bits, max_out_b)

        for configs in itertools.product(agg_spreads, agg_inputs):
            inp_name = configs[1][0]
            seed = configs[1][2]
            spread_name = configs[0][0]

            inp = configs[1][1]
            cfull_tpl = full_tpl

            agg_configs.append((inp, cfull_tpl))
            ename = '%s%s-%s-raw-r%s-inp-%s-spr-%s-s%sMB' \
                    % (eprefix or '', cpos[0], 'bin',
                       cpos[2], inp_name, spread_name, int(max_out / 1024 / 1024))

            lowmc_cfg = {
                "type": "block",
                "init_frequency": "only_once",
                "algorithm": "LOWMC",
                "round": cround,
                "block_size": inp_block_bytes,
                "block_size_bits": inp_block_bytes * 8,
                "key_size_bits": key_size * 8,
                "plaintext": inp['stream'],
                "key_size": key_size,
                "key": {
                    "type": "pcg32_stream"
                },
                "iv_size": key_size,
                "iv": {
                    "type": "false_stream"
                }
            }

            if sboxes:
                lowmc_cfg["sboxes"] = sboxes

            script = {
                "stream": {
                    "type": "shell",
                    "direct_file": False,
                    "exec": cfull_tpl,
                },
                "input_files": {
                    "CONFIG1.JSON": {
                        "note": inp['notes'],
                        "data": {
                            "notes": ename,
                            "seed": seed,
                            "tv_size": inp_block_bytes,
                            "tv_count": tv_count,
                            "file_name": "%s.bin" % ename,
                            "stdout": True,
                            "stream": lowmc_cfg
                        }
                    }
                }
            }
            agg_scripts.append(ExpRec(ename=ename, ssize=max_out / 1024 / 1024, fname=ename + '.json',
                                      tpl_file=script, cfg_type='rtt-data-gen-config'))
    return agg_scripts


def int_to_hex(input, nbytes=1):
    return binascii.hexlify(input.to_bytes(nbytes, byteorder='big')).decode('utf8')


def int_to_seed(seed):
    return int_to_hex(seed, 8)


def gen_col_iv(is_block=True):
    if is_block:
        return {
            "type": "false_stream"
        }
    else:
        return {
            "type": "repeating_stream",
            "period": 1,
            "source": {
                "type": "false_stream"
            }
        }


def generate_block_col(algorithm, data_size, cround=1, tv_size=16, key_size=16, iv_size=0, nexps=3, eprefix='',
                       streams=StreamOptions.CTR_LHW):
    return generate_cfg_col('block', algorithm, data_size, cround, tv_size, key_size, iv_size, nexps, eprefix, streams)


def generate_stream_col(algorithm, data_size, cround=1, tv_size=16, key_size=16, iv_size=0, nexps=3, eprefix='',
                        streams=StreamOptions.CTR_LHW):
    return generate_cfg_col('stream_cipher', algorithm, data_size, cround, tv_size, key_size, iv_size, nexps, eprefix,
                            streams)


def generate_prng_col(algorithm, data_size, cround=1, tv_size=16, key_size=16, iv_size=0, nexps=3, eprefix='',
                      streams=StreamOptions.CTR_LHW):
    return generate_cfg_col('prng', algorithm, data_size, cround, tv_size, key_size, iv_size, nexps, eprefix,
                            streams)


def generate_streams(tv_count, tv_size, streams=StreamOptions.CTR_LHW, nexps=3):
    agg_inputs = []
    # CTR
    for ix in range(nexps if StreamOptions.has_ctr(streams) else 0):
        sscript = make_ctr_config(tv_size, offset=int_to_hex(ix, 1), tv_count=tv_count,
                                  core_only=True)  # type: dict
        agg_inputs.append(
            StreamRec(stype='ctr', sdesc=f'{tv_size * 8}sbit-offset-{ix}', sscript=sscript,
                      expid=ix, seed=int_to_seed(ix))
        )

    # LHW
    weight = comp_hw_weight(tv_size, samples=nexps, min_samples=tv_count)
    for ix in range(nexps if StreamOptions.has_lhw(streams) else 0):
        sscript = make_hw_config(tv_size, weight=weight, offset_range=ix / float(nexps),
                                 tv_count=tv_count, return_aux=True)  # type: HwConfig
        agg_inputs.append(
            StreamRec(stype='hw', sdesc=sscript.note, sscript=sscript.core,
                      expid=ix, seed=int_to_seed(nexps + ix))
        )

    # SAC
    for ix in range(nexps if StreamOptions.has_sac(streams) else 0):
        sscript = {'type': 'sac'}
        agg_inputs.append(
            StreamRec(stype='sac', sdesc=f'{tv_size * 8}sbit-offset-{ix}', sscript=sscript,
                      expid=ix, seed=int_to_seed(2 * nexps + ix))
        )

    # RND
    for ix in range(nexps if StreamOptions.has_rnd(streams) else 0):
        sscript = {'type': 'pcg32_stream'}
        agg_inputs.append(
            StreamRec(stype='rnd', sdesc=f'{tv_size * 8}sbit-offset-{ix}', sscript=sscript,
                      expid=ix, seed=int_to_seed(3 * nexps + ix))
        )
    return agg_inputs


def generate_cfg_col(alg_type, algorithm, data_size, cround=1, tv_size=16, key_size=16, iv_size=0, nexps=3, eprefix='',
                     streams=StreamOptions.CTR_LHW):
    """
    tv_size defines number of bytes to generate using current key value
    Inspired by taro_proc.py
    """
    tv_count = int(math.ceil(data_size / tv_size))
    key_count = tv_count  # number of keys = number of test vectors. Reseed with each TV
    inp_block_bytes = key_size
    size_mbs = int(math.ceil(data_size / 1024 / 1024))

    is_block = alg_type == 'block'
    is_stream = alg_type == 'stream_cipher'
    is_prng = alg_type == 'prng'
    if not is_block and not is_stream and not is_prng:
        raise ValueError('Unknown alg type: %s' % (alg_type,))

    agg_inputs = generate_streams(tv_count=key_count, tv_size=inp_block_bytes, streams=streams, nexps=nexps)
    agg_scripts = []
    for configs in agg_inputs:
        src_type = configs.stype
        inp_name = configs.sdesc
        key_config = configs.sscript
        eid = configs.expid
        seed = configs.seed

        note = f'{eprefix}{algorithm}-t:{alg_type}-r:{cround}-b:{tv_size}-' \
               f's:{size_mbs}MiB-e:{eid}-i:{src_type}.key-{inp_name}'
        fname = note.replace(':', '_') + '.json'

        tpl = {
            "notes": "generated by generator.py",
            "seed": seed,
            "tv-size": None,
            "tv-count": None,
            "tv_size": tv_size,
            "tv_count": tv_count,
            "stdout": True,
            "file_name": "file.bin",
            "stream": {
                "type": alg_type,
                "algorithm": algorithm,
                "round": cround,
                "block_size": tv_size,
                "plaintext": {
                    "type": "false_stream"
                },
                "key_size": key_size,
                "key": key_config,
                "iv_size": iv_size,
                "iv": gen_col_iv(is_block=is_block)
            },
            "note": note
        }

        if is_stream:
            tpl['stream']['generator'] = "pcg32"
        elif is_block:
            tpl['stream']['init_frequency'] = "1"
        elif is_prng:
            tpl['stream'] = {
                "type": "prng",
                "algorithm": algorithm,
                "reseed_for_each_test_vector": True,
                "seeder": key_config
            }

        agg_scripts.append(ExpRec(ename=note, ssize=size_mbs, fname=fname, tpl_file=tpl, cfg_type='cryptostreams-config'))
    return agg_scripts


def generate_prng_inp(algorithm, data_size, cround=1, tv_size=16, key_size=16, iv_size=0, nexps=3, eprefix='',
                      streams=StreamOptions.CTR_LHW_SAC):
    return generate_cfg_inp('prng', algorithm, data_size, cround, tv_size, key_size, iv_size, nexps, eprefix, streams)


def generate_hash_inp(algorithm, data_size, cround=1, tv_size=16, key_size=16, iv_size=0, nexps=3, eprefix='',
                      streams=StreamOptions.CTR_LHW_SAC):
    return generate_cfg_inp('hash', algorithm, data_size, cround, tv_size, key_size, iv_size, nexps, eprefix, streams)


def generate_block_inp(algorithm, data_size, cround=1, tv_size=16, key_size=16, iv_size=0, nexps=3, eprefix='',
                       streams=StreamOptions.CTR_LHW_SAC):
    return generate_cfg_inp('block', algorithm, data_size, cround, tv_size, key_size, iv_size, nexps, eprefix, streams)


def generate_stream_inp(algorithm, data_size, cround=1, tv_size=16, key_size=16, iv_size=0, nexps=3, eprefix='',
                        streams=StreamOptions.CTR_LHW_SAC):
    return generate_cfg_inp('stream_cipher', algorithm, data_size, cround, tv_size, key_size, iv_size, nexps, eprefix,
                            streams)


def generate_cfg_inp(alg_type, algorithm, data_size, cround=1, tv_size=16, key_size=16, iv_size=0, nexps=3, eprefix='',
                     streams=StreamOptions.CTR_LHW_SAC):
    """
    Generates cryptostreams-based hash/block/stream cipher/prng config with plaintext/source input strategies
    """
    tv_count = int(math.ceil(data_size / tv_size))
    size_mbs = int(math.ceil(data_size / 1024 / 1024))

    is_block = alg_type == 'block'
    is_stream = alg_type == 'stream_cipher'
    is_prng = alg_type == 'prng'
    is_hash = alg_type == 'hash'
    if not is_block and not is_stream and not is_prng and not is_hash:
        raise ValueError('Unknown alg type: %s' % (alg_type,))

    agg_inputs = generate_streams(tv_count=tv_count, tv_size=tv_size, streams=streams, nexps=nexps)
    agg_scripts = []
    for configs in agg_inputs:
        src_type = configs.stype
        inp_name = configs.sdesc
        inp_config = configs.sscript
        eid = configs.expid
        seed = configs.seed

        note = f'{eprefix}{algorithm}-t:{alg_type}-r:{cround}-b:{tv_size}-' \
               f's:{size_mbs}MiB-e:{eid}-i:{src_type}.key-{inp_name}'
        fname = note.replace(':', '_') + '.json'

        tpl = {
            "notes": "generated by generator.py",
            "seed": seed,
            "tv-size": None,
            "tv-count": None,
            "tv_size": tv_size,
            "tv_count": tv_count,
            "stdout": True,
            "file_name": "file.bin",
            "stream": {
                "type": alg_type,
                "algorithm": algorithm,
                "round": cround,
                "block_size": tv_size,
            },
            "note": note
        }

        if is_hash:
            tpl['stream']['source'] = inp_config
            tpl['stream']['hash_size'] = tv_size
            tpl['stream']['input_size'] = tv_size

        elif is_prng:
            tpl['stream'] = {
                "type": "prng",
                "algorithm": algorithm,
                "reseed_for_each_test_vector": True,
                "seeder": inp_config
            }

        else:
            tpl['stream']['plaintext'] = inp_config
            tpl['stream']['key_size'] = key_size
            tpl['stream']['key'] = {
                "type": "pcg32_stream"
            }
            tpl['stream']['iv_size'] = iv_size
            tpl['stream']['iv'] = {
                "type": "false_stream"
            }

        if is_stream or is_hash:
            tpl['stream']['generator'] = "pcg32"

        agg_scripts.append(ExpRec(ename=note, ssize=size_mbs, fname=fname, tpl_file=tpl, cfg_type='cryptostreams-config'))
    return agg_scripts


def write_submit(data, cfg_type='rtt-data-gen-config'):
    ndata = [
        ExpRec(ename=x[0], ssize=int(x[1]), fname='%s.json' % x[0], tpl_file=x[2], cfg_type=cfg_type) for x in data
    ]
    return write_submit_obj(ndata)


def write_submit_obj(data: List[ExpRec]):
    with open('__enqueue.sh', 'w+') as fh:
        fh.write("#!/bin/bash\n")

        for ix, coff in enumerate(data):
            with open(coff.fname, 'w+') as fhc:
                fhc.write(json.dumps(coff.tpl_file, indent=2))

            # fh.write("submit_experiment --dieharder --email ph4r05@gmail.com  --name 't06-dieharder-%s' --cfg 'dieharder-paper-1GB.json' --cryptostreams-config '%s'\n" % (name, name))
            fh.write(f"echo '{ix+1}/{len(data)}'\n")
            fh.write("submit_experiment --all_batteries "
                     "--name '%s' "
                     "--cfg '/home/debian/rtt-home/RTTWebInterface/media/predefined_configurations/%sMB.json' "
                     "--%s '%s'\n" % (coff.ename, int(coff.ssize), coff.cfg_type, coff.fname))
