#!/usr/bin/python3

import json
from copy import deepcopy
from generator_defaults import CfgDefaults, FunArgs


class GeneratorCfgSAC:
    cs_input_strategy_prefix = 'sac'
    plaintext_target_stream = CfgDefaults.sac_stream

    # used funs in batch
    stream_cipher_funs = {}
    stream_cipher_default = {
        'type': 'stream_cipher',
        'generator': 'pcg32',
        'algorithm': None,
        'round': None,
        'block_size': None,
        'plaintext': plaintext_target_stream,
        'key_size': None,
        'key': CfgDefaults.random_stream,
        'iv_size': None,
        'iv': CfgDefaults.false_stream
    }

    hash_funs = {
        'ARIRANG': FunArgs(32, None, None, range(3, 6)),
        'BLAKE': FunArgs(32, None, None, (1, 2, 3)),
        'Boole': FunArgs(32, None, None, range(1, 6)),
        'Cheetah': FunArgs(32, None, None, range(3, 7)),
        'DCH': FunArgs(32, None, None, range(1, 5)),
        'DynamicSHA2': FunArgs(32, None, None, range(10, 15)),
        'ECHO': FunArgs(32, None, None, range(1, 9)),
        'Gost': FunArgs(32, None, None, range(1, 6)),
        #'Grostl': FunArgs(32, None, None, (1, 2)),
        'Hamsi': FunArgs(32, None, None, range(1, 4)),
        'JH': FunArgs(32, None, None, (5, 6, 7, 8)),
        'Keccak': FunArgs(32, None, None, (1, 2, 3)),
        'Lesamnta': FunArgs(32, None, None, range(1, 6)),
        'Luffa': FunArgs(32, None, None, range(6, 9)),
        'MD5': FunArgs(16, None, None, (9, 10, 11, 12, 13, 14, 15)),
        'MD6': FunArgs(32, None, None, (5, 6, 7, 8, 9)),
        'RIPEMD160': FunArgs(20, None, None, (8, 9, 10, 11, 12, 13, 14, 15)),
        'SHA1': FunArgs(20, None, None, (11, 12, 13, 14, 15, 16, 17, 18)),
        'SHA2': FunArgs(32, None, None, (10, 11, 12, 13)),
        'SHA3': FunArgs(32, None, None, range(1, 6)),
        'Skein': FunArgs(32, None, None, (2, 3, 4)),
        'Tangle': FunArgs(32, None, None, range(18, 25)),
        'Tangle2': FunArgs(32, None, None, range(18, 25)),
        'Tiger': FunArgs(24, None, None, (1, 2)),
        'Twister': FunArgs(32, None, None, range(6, 10)),
        'Whirlpool': FunArgs(64, None, None, (2, 4))
    }
    hash_default = {
        'type': 'hash',
        'generator': 'pcg32',
        'algorithm': None,
        'round': None,
        'hash_size': None,
        'input_size': None,
        'source': plaintext_target_stream
    }

    block_funs = {
        'AES': FunArgs(16, 16, None, (1, 2, 3, 4)),
        'ARIA': FunArgs(16, 16, None, (1, 2, 3)),
        'BLOWFISH': FunArgs(8, 32, None, (1, 2, 3)),
        'CAMELLIA': FunArgs(16, 16, None, (2, 3, 4, 5)),
        'CAST': FunArgs(8, 16, None, (2, 3, 4, 5)),
        'CHASKEY': FunArgs(16, 16, None, (1, 2, 3, 4)),
        'FANTOMAS': FunArgs(16, 16, None, (1, 2, 3, 4)),
        'GOST': FunArgs(8, 32, None, (5, 6, 7, 8, 9)),
        'HIGHT': FunArgs(8, 16, None, (7, 8, 9, 10)),
        'IDEA': FunArgs(8, 16, None, (1, 2, 3)),
        'KASUMI': FunArgs(8, 16, None, (1, 2, 3, 4, 5)),
        'KUZNYECHIK': FunArgs(16, 32, None, (1, 2, 3)),
        'LBLOCK': FunArgs(8, 10, None, (7, 8, 9, 10, 11)),
        'LEA': FunArgs(16, 16, None, (4, 5, 6, 7, 8, 9, 10)),
        'LED': FunArgs(8, 10, None, (2, 3, 4, 5)),
        'MARS': FunArgs(16, 16, None, (0, 1)),
        'MISTY1': FunArgs(8, 16, None, (1, 2, 3)),
        'NOEKEON': FunArgs(16, 16, None, (1, 2, 3, 4)),
        'PICCOLO': FunArgs(8, 10, None, (2, 3, 4, 5)),
        'PRIDE': FunArgs(8, 16, None, (3, 4, 5, 6, 7)),
        'PRINCE': FunArgs(8, 16, None, (2, 3, 4, 5)),
        'RC5-20': FunArgs(8, 16, None, (2, 3, 4, 5, 6, 7)),
        'RC6': FunArgs(16, 16, None, (1, 2, 3, 4)),
        'RECTANGLE-K80': FunArgs(8, 10, None, range(4, 10)),
        'RECTANGLE-K128': FunArgs(8, 16, None, range(4, 10)),
        'ROAD-RUNNER-K80': FunArgs(8, 10, None, range(1, 5)),
        'ROAD-RUNNER-K128': FunArgs(8, 16, None, range(1, 5)),
        'ROBIN': FunArgs(16, 16, None, (1, 2, 3, 4)),
        'ROBIN-STAR': FunArgs(16, 16, None, range(1, 5)),
        'SEED': FunArgs(16, 16, None, (1, 2, 3, 4)),
        'SERPENT': FunArgs(16, 16, None, (1, 2, 3, 4)),
        'SHACAL2': FunArgs(32, 64, None, (2, 3, 4, 5, 6, 7)),
        'SIMON': FunArgs(16, 16, None, (12, 13, 14, 15, 16, 17, 18, 19)),
        'SINGLE-DES': FunArgs(8, 7, None, (3, 4, 5, 6)),
        'SPARX-B64': FunArgs(8, 16, None, range(1, 5)),
        'SPARX-B128': FunArgs(16, 16, None, range(1, 5)),
        'SPECK': FunArgs(16, 16, None, (5, 6, 7, 8)),
        'TEA': FunArgs(8, 16, None, (2, 3, 4, 5)),
        'TRIPLE-DES': FunArgs(8, 21, None, (1, 2, 3)),
        'TWINE': FunArgs(8, 10, None, (6, 7, 8, 9)),
        'TWOFISH': FunArgs(16, 16, None, (1, 2, 3, 4)),
        'XTEA': FunArgs(8, 16, None, (1, 2, 3, 4, 5)),
    }
    block_default = {
        'type': 'block',
        'init_frequency': 'only_once',
        'algorithm': None,
        'round': None,
        'block_size': 16,
        'plaintext': plaintext_target_stream,
        'key_size': 16,
        'key': CfgDefaults.random_stream,
        'iv_size': 16,
        'iv': CfgDefaults.false_stream
    }

    def prepare_cfg(self, project, fun, rounds, tv_size, tv_num, seed, name_prefix):
        cfg_name = '{}_{}_r{:02d}_b{}.json'.format(name_prefix, fun, rounds, tv_size)
        bin_name = '{}_{}_r{:02d}_b{}.bin'.format(name_prefix, fun, rounds, tv_size)

        with open(cfg_name, 'w') as f:

            current_cfg = deepcopy(CfgDefaults.config_base)
            current_cfg['seed'] = seed
            current_cfg['tv_size'] = tv_size
            current_cfg['tv_count'] = tv_num
            current_cfg['file_name'] = bin_name

            if project == "stream_cipher":
                stream = deepcopy(self.stream_cipher_default)
                stream['algorithm'] = fun
                stream['round'] = rounds
                stream['block_size'] = self.stream_cipher_funs[fun].block_size
                stream['key_size'] = self.stream_cipher_funs[fun].key_size
                stream['iv_size'] = self.stream_cipher_funs[fun].iv_size
                current_cfg['stream'] = stream

            elif project == "hash":
                stream = deepcopy(self.hash_default)
                stream['algorithm'] = fun
                stream['round'] = rounds
                stream['hash_size'] = self.hash_funs[fun].block_size
                stream['input_size'] = self.hash_funs[fun].block_size
                current_cfg['stream'] = stream			
				
            elif project == "block":
                stream = deepcopy(self.block_default)
                stream['algorithm'] = fun
                stream['round'] = rounds
                stream['block_size'] = self.block_funs[fun].block_size
                stream['key_size'] = self.block_funs[fun].key_size
                current_cfg['stream'] = stream

            else:  # rnd
                stream = deepcopy(CfgDefaults.random_stream)
                stream['algorithm'] = fun
                stream['round'] = 0
                stream['block_size'] = 16
                current_cfg['stream'] = stream

            f.write(json.dumps(current_cfg))
            f.close()

        return cfg_name
