from typing import Optional


class FuncInfo:
    HASH = 1
    CIPHER = 2
    PRNG = 3

    NONE = 0
    BLOCK = 1
    STREAM = 2
    LIGHT = 3
    MPC = 4

    def __init__(self, fname, ftype, stype=None, max_rounds=None, human_broken_rounds=None, partial_rounds=None,
                 block_size=None, key_size=None, iv_size=None):
        self.fname = fname
        self.ftype = ftype
        self.stype = stype or 0
        self.max_rounds = max_rounds
        self.partial_rounds = partial_rounds
        self.human_broken_rounds = human_broken_rounds
        self.block_size = block_size
        self.key_size = key_size
        self.iv_size = iv_size

    def get_alg_type(self):
        return get_alg_type(self.ftype, self.stype)

    def __repr__(self):
        return f'FuncInfo({self.fname}, ft: {self.ftype}, st: {self.stype}, r: {self.max_rounds})'

    @staticmethod
    def from_str(x):
        if x in ('block', 'stream_cipher'):
            return FuncInfo.CIPHER
        elif x == 'prng':
            return FuncInfo.PRNG
        elif x == 'hash':
            return FuncInfo.HASH
        else:
            raise ValueError(f'Unknown type: {x}')


class FuncDb:
    def __init__(self):
        self.fncs = []
        self.recs = {}

    def _key(self, fname, ftype=FuncInfo.CIPHER):
        return '%s;%s' % (fname.lower(), ftype if ftype is not None else '')

    def add(self, e: FuncInfo):
        k = self._key(e.fname, e.ftype)
        if e in self.recs:
            raise ValueError('Duplicate entry: %s' % k)

        self.fncs.append(e)
        self.recs[k] = e

        k2 = self._key(e.fname, None)
        self.recs[k2] = e

    def __iadd__(self, other):
        self.add(other)
        return self

    def add_all(self, collection):
        for x in collection:
            self.add(x)

    def search(self, fname, ftype=None, return_none=True) -> Optional[FuncInfo]:
        k = self._key(fname, ftype)
        if k not in self.recs:
            if return_none:
                return None
            raise ValueError('Function not present %s' % k)
        return self.recs[k]


def get_alg_type(ftype, stype):
    if ftype == FuncInfo.HASH:
        return 'hash'
    elif ftype == FuncInfo.PRNG:
        return 'prng'
    elif stype == FuncInfo.STREAM:
        return 'stream_cipher'
    elif ftype == FuncInfo.CIPHER and stype in [FuncInfo.BLOCK, FuncInfo.NONE, None]:
        return 'block'
    else:
        raise ValueError(f'Unknown algtype for {ftype}:{stype}')


FUNC_DB = FuncDb()
FUNC_DB.add_all([
    FuncInfo('ABC', FuncInfo.HASH, None, 9, None),  # v3, https://eprint.iacr.org/2010/658.pdf
    FuncInfo('Achterbahn', FuncInfo.HASH, None, 1, None),  # https://en.wikipedia.org/wiki/Achterbahn no rounds
    FuncInfo('DICING', FuncInfo.HASH, None, None, None),
    FuncInfo('Dragon', FuncInfo.HASH, None, None, None),  # https://www.ecrypt.eu.org/stream/p3ciphers/dragon/dragon_p3.pdf

    FuncInfo('AES', FuncInfo.CIPHER, None, 10, 6, block_size=16, key_size=16, iv_size=16),
    FuncInfo('ARIA', FuncInfo.CIPHER, None, 12, 4, block_size=16, key_size=16, iv_size=16),
    FuncInfo('Blowfish', FuncInfo.CIPHER, None, 16, 4, block_size=8, key_size=32, iv_size=16),
    FuncInfo('Camelia', FuncInfo.CIPHER, None, 18, 8, block_size=16, key_size=16, iv_size=16),
    FuncInfo('Camellia', FuncInfo.CIPHER, None, 18, 8, block_size=16, key_size=16, iv_size=16),
    FuncInfo('Cast', FuncInfo.CIPHER, None, 12, 9, block_size=8, key_size=16, iv_size=16),
    FuncInfo('Chaskey', FuncInfo.CIPHER, None, 16, 7, block_size=16, key_size=16, iv_size=16),
    FuncInfo('Fantomas', FuncInfo.CIPHER, None, 12, 4, block_size=16, key_size=16, iv_size=16),
    FuncInfo('Gost', FuncInfo.CIPHER, None, 32, None, block_size=8, key_size=32, iv_size=16),
    FuncInfo('Hight', FuncInfo.CIPHER, None, 32, None, block_size=8, key_size=16, iv_size=16),
    FuncInfo('Idea', FuncInfo.CIPHER, None, 8, 4, block_size=8, key_size=16, iv_size=16),
    FuncInfo('Kasumi', FuncInfo.CIPHER, None, 8, 8, block_size=8, key_size=16, iv_size=16),
    FuncInfo('KUZNYECHIK', FuncInfo.CIPHER, None, 10, 4, block_size=16, key_size=32, iv_size=16),
    FuncInfo('LBLOCK', FuncInfo.CIPHER, None, 32, 24, block_size=8, key_size=10, iv_size=16),
    FuncInfo('LEA', FuncInfo.CIPHER, None, 24, None, block_size=16, key_size=16, iv_size=16),
    FuncInfo('LED', FuncInfo.CIPHER, None, 48, None, block_size=8, key_size=10, iv_size=16),
    FuncInfo('MARS', FuncInfo.CIPHER, None, 16, None, block_size=16, key_size=16, iv_size=16),
    FuncInfo('MISTY1', FuncInfo.CIPHER, None, 8, 6, block_size=8, key_size=16, iv_size=16),
    FuncInfo('NOEKEON', FuncInfo.CIPHER, None, 16, 4, block_size=16, key_size=16, iv_size=16),
    FuncInfo('PICCOLO', FuncInfo.CIPHER, None, 25, None, block_size=8, key_size=10, iv_size=16),
    FuncInfo('PRIDE', FuncInfo.CIPHER, None, 20, 19, block_size=8, key_size=16, iv_size=16),
    FuncInfo('Prince', FuncInfo.CIPHER, None, 12, 6, block_size=8, key_size=16, iv_size=16),
    FuncInfo('RC5-20', FuncInfo.CIPHER, None, 20, None, block_size=8, key_size=16, iv_size=16),
    FuncInfo('RC6', FuncInfo.CIPHER, None, 20, None, block_size=16, key_size=16, iv_size=16),
    FuncInfo('RECTANGLE-K128', FuncInfo.CIPHER, None, 25, 14, block_size=8, key_size=16, iv_size=16),
    FuncInfo('RECTANGLE-K80', FuncInfo.CIPHER, None, 25, 18, block_size=8, key_size=10, iv_size=16),
    FuncInfo('ROAD-RUNNER-K128', FuncInfo.CIPHER, None, 12, None, block_size=8, key_size=16, iv_size=16),
    FuncInfo('ROAD-RUNNER-K80', FuncInfo.CIPHER, None, 10, None, block_size=8, key_size=10, iv_size=16),
    FuncInfo('ROBIN', FuncInfo.CIPHER, None, 16, 16, block_size=16, key_size=16, iv_size=16),
    FuncInfo('ROBIN-STAR', FuncInfo.CIPHER, None, 16, None, block_size=16, key_size=16, iv_size=16),
    FuncInfo('SEED', FuncInfo.CIPHER, None, 16, None, block_size=16, key_size=16, iv_size=16),
    FuncInfo('SERPENT', FuncInfo.CIPHER, None, 32, 5, block_size=16, key_size=16, iv_size=16),
    FuncInfo('SHACAL2', FuncInfo.CIPHER, None, 80, None, block_size=32, key_size=64, iv_size=16),
    FuncInfo('SIMON', FuncInfo.CIPHER, None, 68, 17, block_size=16, key_size=16, iv_size=16),
    FuncInfo('SINGLE-DES', FuncInfo.CIPHER, None, 16, 16, block_size=8, key_size=7, iv_size=16),
    FuncInfo('SPARX-B128', FuncInfo.CIPHER, None, 32, 8, block_size=16, key_size=16, iv_size=16),
    FuncInfo('SPARX-B64', FuncInfo.CIPHER, None, 24, 8, block_size=8, key_size=16, iv_size=16),
    FuncInfo('SPECK', FuncInfo.CIPHER, None, 32, 15, block_size=16, key_size=16, iv_size=16),
    FuncInfo('TEA', FuncInfo.CIPHER, None, 32, 5, block_size=8, key_size=16, iv_size=16),
    FuncInfo('TRIPLE-DES', FuncInfo.CIPHER, None, 16, None, block_size=8, key_size=21, iv_size=16),
    FuncInfo('TWINE', FuncInfo.CIPHER, None, 35, 23, block_size=8, key_size=10, iv_size=16),
    FuncInfo('TWOFISH', FuncInfo.CIPHER, None, 16, 16, block_size=16, key_size=16, iv_size=16),
    FuncInfo('XTEA', FuncInfo.CIPHER, None, 32, 8, block_size=8, key_size=16, iv_size=16),

    FuncInfo('ARIRANG', FuncInfo.HASH, None, 4, 4, block_size=32),
    FuncInfo('AURORA', FuncInfo.HASH, None, 17, None, block_size=32),
    FuncInfo('Abacus', FuncInfo.HASH, None, 280, None, block_size=32),
    FuncInfo('BLAKE', FuncInfo.HASH, None, 14, 4, block_size=32),
    FuncInfo('BMW', FuncInfo.HASH, None, 16, None, block_size=32),
    FuncInfo('Blender', FuncInfo.HASH, None, 32, None, block_size=64),
    FuncInfo('Boole', FuncInfo.HASH, None, 16, 16, block_size=32),
    FuncInfo('CHI', FuncInfo.HASH, None, 20, None, block_size=64),
    FuncInfo('Cheetah', FuncInfo.HASH, None, 16, 12, block_size=32),
    FuncInfo('CubeHash', FuncInfo.HASH, None, 8, None, block_size=32),
    FuncInfo('DCH', FuncInfo.HASH, None, 4, 4, block_size=32),
    FuncInfo('DynamicSHA', FuncInfo.HASH, None, 16, None, block_size=48),
    FuncInfo('DynamicSHA2', FuncInfo.HASH, None, 17, 17, block_size=32),
    FuncInfo('ECHO', FuncInfo.HASH, None, 8, 4, block_size=32),
    FuncInfo('ESSENCE', FuncInfo.HASH, None, 32, None, block_size=32),
    FuncInfo('Gost', FuncInfo.HASH, None, 10, 5, block_size=32),
    FuncInfo('Grostl', FuncInfo.HASH, None, 10, None, block_size=32),
    FuncInfo('Hamsi', FuncInfo.HASH, None, 3, None, block_size=32),
    FuncInfo('JH', FuncInfo.HASH, None, 42, 10, block_size=32),
    FuncInfo('Keccak', FuncInfo.HASH, None, 24, 5, block_size=32),
    FuncInfo('Lesamnta', FuncInfo.HASH, None, 32, 32, block_size=32),
    FuncInfo('Luffa', FuncInfo.HASH, None, 8, 8, block_size=32),
    FuncInfo('MCSSHA3', FuncInfo.HASH, None, 1, None, block_size=32),
    FuncInfo('MD5', FuncInfo.HASH, None, 64, 64, block_size=16),
    FuncInfo('MD6', FuncInfo.HASH, None, 104, 16, block_size=32),
    FuncInfo('RIPEMD160', FuncInfo.HASH, None, 80, 48, block_size=20),
    FuncInfo('SHA1', FuncInfo.HASH, None, 80, 80, block_size=20),
    FuncInfo('SHA2', FuncInfo.HASH, None, 64, 31, block_size=32),
    FuncInfo('SHA3', FuncInfo.HASH, None, 24, 5, block_size=32),
    FuncInfo('SHAvite3', FuncInfo.HASH, None, 12, None, block_size=32),
    FuncInfo('SIMD', FuncInfo.HASH, None, 4, None, block_size=32),
    FuncInfo('Sarmal', FuncInfo.HASH, None, 16, None, block_size=32),
    FuncInfo('Shabal', FuncInfo.HASH, None, 1, None, block_size=32),
    FuncInfo('Skein', FuncInfo.HASH, None, 72, 17, block_size=32),
    FuncInfo('TIB3', FuncInfo.HASH, None, 16, None, block_size=48),
    FuncInfo('Tangle', FuncInfo.HASH, None, 80, 80, block_size=32),
    FuncInfo('Tangle2', FuncInfo.HASH, None, 80, None, block_size=32),
    FuncInfo('Tiger', FuncInfo.HASH, None, 23, 19, block_size=24),
    FuncInfo('Twister', FuncInfo.HASH, None, 9, 9, block_size=32),
    FuncInfo('Whirlpool', FuncInfo.HASH, None, 10, 6, block_size=64),

    FuncInfo('Chacha', FuncInfo.CIPHER, FuncInfo.STREAM, 20, None, block_size=32, key_size=32, iv_size=8),
    FuncInfo('DECIM', FuncInfo.CIPHER, FuncInfo.STREAM, 8, None, block_size=24, key_size=10, iv_size=8),
    FuncInfo('F-FCSR', FuncInfo.CIPHER, FuncInfo.STREAM, 5, None, block_size=16, key_size=16, iv_size=8),
    FuncInfo('Fubuki', FuncInfo.CIPHER, FuncInfo.STREAM, 4, None, block_size=16, key_size=16, iv_size=16),
    FuncInfo('Grain', FuncInfo.CIPHER, FuncInfo.STREAM, 13, None, block_size=16, key_size=16, iv_size=12),
    FuncInfo('HC-128', FuncInfo.CIPHER, FuncInfo.STREAM, 1, None, block_size=16, key_size=16, iv_size=16),
    FuncInfo('Hermes', FuncInfo.CIPHER, FuncInfo.STREAM, 10, None, block_size=16, key_size=10, iv_size=0),
    FuncInfo('LEX', FuncInfo.CIPHER, FuncInfo.STREAM, 10, None, block_size=16, key_size=16, iv_size=16),
    FuncInfo('MICKEY', FuncInfo.CIPHER, FuncInfo.STREAM, 1, None, block_size=16, key_size=16, iv_size=0),
    FuncInfo('RC4', FuncInfo.CIPHER, FuncInfo.STREAM, 1, None, block_size=32, key_size=16, iv_size=0),
    FuncInfo('Rabbit', FuncInfo.CIPHER, FuncInfo.STREAM, 4, None, block_size=16, key_size=16, iv_size=8),
    FuncInfo('SOSEMANUK', FuncInfo.CIPHER, FuncInfo.STREAM, 25, None, block_size=16, key_size=16, iv_size=16),
    FuncInfo('Salsa20', FuncInfo.CIPHER, FuncInfo.STREAM, 20, None, block_size=8, key_size=16, iv_size=8),
    FuncInfo('TSC-4', FuncInfo.CIPHER, FuncInfo.STREAM, 32, None, block_size=32, key_size=10, iv_size=10),
    FuncInfo('Trivium', FuncInfo.CIPHER, FuncInfo.STREAM, 9, None, block_size=8, key_size=10, iv_size=10),

    FuncInfo('gmimc-S45a', FuncInfo.CIPHER, FuncInfo.MPC, 121),
    FuncInfo('gmimc-S45b', FuncInfo.CIPHER, FuncInfo.MPC, 137),
    FuncInfo('gmimc-S80a', FuncInfo.CIPHER, FuncInfo.MPC, 111),
    FuncInfo('gmimc-S80b', FuncInfo.CIPHER, FuncInfo.MPC, 210),
    FuncInfo('gmimc-S80c', FuncInfo.CIPHER, FuncInfo.MPC, 226),
    FuncInfo('gmimc-S128a', FuncInfo.CIPHER, FuncInfo.MPC, 166),
    FuncInfo('gmimc-S128b', FuncInfo.CIPHER, FuncInfo.MPC, 166),
    FuncInfo('gmimc-S128c', FuncInfo.CIPHER, FuncInfo.MPC, 182),
    FuncInfo('gmimc-S128d', FuncInfo.CIPHER, FuncInfo.MPC, 101),
    FuncInfo('gmimc-S128e', FuncInfo.CIPHER, FuncInfo.MPC, 342),
    FuncInfo('gmimc-S256f', FuncInfo.CIPHER, FuncInfo.MPC, 174),
    FuncInfo('gmimc-S256b', FuncInfo.CIPHER, FuncInfo.MPC, 186),

    FuncInfo('mimc_hash-S45', FuncInfo.CIPHER, FuncInfo.MPC, 116),
    FuncInfo('mimc_hash-S80', FuncInfo.CIPHER, FuncInfo.MPC, 204),
    FuncInfo('mimc_hash-S128', FuncInfo.CIPHER, FuncInfo.MPC, 320),

    FuncInfo('lowmc-s80a', FuncInfo.CIPHER, FuncInfo.MPC, 12),
    FuncInfo('lowmc-s80b', FuncInfo.CIPHER, FuncInfo.MPC, 12),
    FuncInfo('lowmc-s128a', FuncInfo.CIPHER, FuncInfo.MPC, 14),
    FuncInfo('lowmc-s128b', FuncInfo.CIPHER, FuncInfo.MPC, 252),
    FuncInfo('lowmc-s128c', FuncInfo.CIPHER, FuncInfo.MPC, 128),
    FuncInfo('lowmc-s128d', FuncInfo.CIPHER, FuncInfo.MPC, 88),

    FuncInfo('Poseidon_S45a', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=26),
    FuncInfo('Starkad_S45a', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=28),
    FuncInfo('Poseidon_S45b', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=28),
    FuncInfo('Starkad_S45b', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=31),
    FuncInfo('Poseidon_S80a', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=51),
    FuncInfo('Starkad_S80a', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=53),
    FuncInfo('Poseidon_S80b', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=50),
    FuncInfo('Starkad_S80b', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=52),
    FuncInfo('Poseidon_S80c', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=52),
    FuncInfo('Starkad_S80c', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=54),
    FuncInfo('Poseidon_S128a', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=81),
    FuncInfo('Starkad_S128a', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=85),
    FuncInfo('Poseidon_S128b', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=83),
    FuncInfo('Starkad_S128b', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=85),
    FuncInfo('Poseidon_S128c', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=83),
    FuncInfo('Starkad_S128c', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=86),
    FuncInfo('Poseidon_S128d', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=40),
    FuncInfo('Starkad_S128d', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=43),
    FuncInfo('Poseidon_S128e', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=85),
    FuncInfo('Starkad_S128e', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=88),
    FuncInfo('Poseidon_S256a', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=82),
    FuncInfo('Starkad_S256a', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=86),
    FuncInfo('Poseidon_S256b', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=83),
    FuncInfo('Starkad_S256b', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=86),
    FuncInfo('Poseidon_S128_BLS12_138', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=60),

    FuncInfo('Rescue_S45a', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),
    FuncInfo('Vision_S45a', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),
    FuncInfo('Rescue_S45b', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),
    FuncInfo('Vision_S45b', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),
    FuncInfo('Rescue_S80a', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),
    FuncInfo('Vision_S80a', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),
    FuncInfo('Rescue_S80b', FuncInfo.CIPHER, FuncInfo.MPC, 14, None),
    FuncInfo('Vision_S80b', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),
    FuncInfo('Rescue_S80c', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),
    FuncInfo('Vision_S80c', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),
    FuncInfo('Rescue_S128a', FuncInfo.CIPHER, FuncInfo.MPC, 16, None),
    FuncInfo('Vision_S128a', FuncInfo.CIPHER, FuncInfo.MPC, 12, None),
    FuncInfo('Rescue_S128b', FuncInfo.CIPHER, FuncInfo.MPC, 22, None),
    FuncInfo('Vision_S128b', FuncInfo.CIPHER, FuncInfo.MPC, 16, None),
    FuncInfo('Rescue_S128c', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),
    FuncInfo('Vision_S128c', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),
    FuncInfo('Rescue_S128d', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),
    FuncInfo('Vision_S128d', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),
    FuncInfo('Rescue_S128e', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),
    FuncInfo('Vision_S128e', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),
    FuncInfo('Rescue_S256a', FuncInfo.CIPHER, FuncInfo.MPC, 16, None),
    FuncInfo('Vision_S256a', FuncInfo.CIPHER, FuncInfo.MPC, 12, None),
    FuncInfo('Rescue_S256b', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),
    FuncInfo('Vision_S256b', FuncInfo.CIPHER, FuncInfo.MPC, 10, None),

    FuncInfo('std_mersenne_twister', FuncInfo.PRNG, max_rounds=0, key_size=4, block_size=4),
    FuncInfo('std_lcg', FuncInfo.PRNG, max_rounds=0, key_size=4, block_size=4),
    FuncInfo('std_subtract_with_carry', FuncInfo.PRNG, max_rounds=0, key_size=4, block_size=4),
    FuncInfo('testu01-ulcg', FuncInfo.PRNG, max_rounds=0, key_size=6, block_size=7),
    FuncInfo('testu01-umrg', FuncInfo.PRNG, max_rounds=0, key_size=6, block_size=7),
    FuncInfo('testu01-uxorshift', FuncInfo.PRNG, max_rounds=0, key_size=32, block_size=4),

])
