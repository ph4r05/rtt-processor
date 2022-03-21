from typing import Optional


class FConstType:
    SPN = 1
    FN = 2    # Feistel
    GFN = 3   # gen Feistel
    ARX = 4  # add–rotate–XOR
    NLFSR = 5
    HYBRID = 6
    LMS = 7
    HASH = 8

    MPRENEL = 1
    MERKLED = 2
    SPONGE = 3


class FuncInfo:
    HASH = 1
    CIPHER = 2
    PRNG = 3

    NONE = 0
    BLOCK = 1
    STREAM = 2
    MPC = 4
    LIGHT = 8

    def __init__(self, fname, ftype, stype=None, max_rounds=None, human_broken_rounds=None, partial_rounds=None,
                 block_size=None, key_size=None, iv_size=None, hash_size=None, year=None, ctype=None):
        self.fname = fname
        self.ftype = ftype
        self.stype = stype or 0
        self.max_rounds = max_rounds
        self.partial_rounds = partial_rounds
        self.human_broken_rounds = human_broken_rounds
        self.block_size = block_size
        self.hash_size = hash_size
        self.key_size = key_size
        self.iv_size = iv_size
        self.year = year
        self.ctype = ctype

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
    elif ftype == FuncInfo.CIPHER and stype in [FuncInfo.BLOCK, FuncInfo.LIGHT, FuncInfo.NONE, None]:
        return 'block'
    else:
        raise ValueError(f'Unknown algtype for {ftype}:{stype}')


FUNC_DB = FuncDb()
FUNC_DB.add_all([
    FuncInfo('ABC', FuncInfo.HASH, None, 9, None),  # v3, https://eprint.iacr.org/2010/658.pdf
    FuncInfo('Achterbahn', FuncInfo.HASH, None, 1, None),  # https://en.wikipedia.org/wiki/Achterbahn no rounds
    FuncInfo('DICING', FuncInfo.HASH, None, None, None),
    FuncInfo('Dragon', FuncInfo.HASH, None, None, None),  # https://www.ecrypt.eu.org/stream/p3ciphers/dragon/dragon_p3.pdf

    FuncInfo('AES', FuncInfo.CIPHER, None, 10, 6, block_size=16, key_size=16, iv_size=16, year=2001, ctype=FConstType.SPN),
    FuncInfo('ARIA', FuncInfo.CIPHER, None, 12, 4, block_size=16, key_size=16, iv_size=16, year=2003, ctype=FConstType.SPN),  # https://en.wikipedia.org/wiki/ARIA_(cipher)
    FuncInfo('Blowfish', FuncInfo.CIPHER, None, 16, 4, block_size=8, key_size=32, iv_size=16, year=1993, ctype=FConstType.FN),
    FuncInfo('Camelia', FuncInfo.CIPHER, None, 18, 8, block_size=16, key_size=16, iv_size=16, year=2000, ctype=FConstType.FN),
    FuncInfo('Camellia', FuncInfo.CIPHER, None, 18, 8, block_size=16, key_size=16, iv_size=16, year=2000, ctype=FConstType.FN),
    FuncInfo('Cast', FuncInfo.CIPHER, None, 12, 9, block_size=8, key_size=16, iv_size=16, year=1996, ctype=FConstType.FN),  # https://en.wikipedia.org/wiki/CAST-128
    FuncInfo('Chaskey', FuncInfo.CIPHER, FuncInfo.LIGHT, 16, 7, block_size=16, key_size=16, iv_size=16, year=2014, ctype=FConstType.ARX),  # https://mouha.be/chaskey/
    FuncInfo('Fantomas', FuncInfo.CIPHER, None, 12, 5, block_size=16, key_size=16, iv_size=16, year=2014, ctype=FConstType.SPN),  # https://www.mdpi.com/2410-387X/3/1/4/htm https://who.paris.inria.fr/Gaetan.Leurent/files/LS_FSE14.pdf
    FuncInfo('Gost', FuncInfo.CIPHER, None, 32, None, block_size=8, key_size=32, iv_size=16, year=1994, ctype=FConstType.FN),  # https://cryptography.fandom.com/wiki/GOST_(block_cipher)
    FuncInfo('Hight', FuncInfo.CIPHER, FuncInfo.LIGHT, 32, None, block_size=8, key_size=16, iv_size=16, year=2006, ctype=FConstType.FN),  # https://link.springer.com/chapter/10.1007/11894063_4
    FuncInfo('Idea', FuncInfo.CIPHER, None, 8, 4, block_size=8, key_size=16, iv_size=16, year=1991, ctype=FConstType.LMS),  # https://en.wikipedia.org/wiki/International_Data_Encryption_Algorithm
    FuncInfo('Kasumi', FuncInfo.CIPHER, None, 8, 8, block_size=8, key_size=16, iv_size=16, year=2005, ctype=FConstType.FN),
    FuncInfo('KUZNYECHIK', FuncInfo.CIPHER, None, 10, 4, block_size=16, key_size=32, iv_size=16, year=2015, ctype=FConstType.SPN),  # https://en.wikipedia.org/wiki/Kuznyechik
    FuncInfo('LBLOCK', FuncInfo.CIPHER, None, 32, 24, block_size=8, key_size=10, iv_size=16, year=2011, ctype=FConstType.FN),
    FuncInfo('LEA', FuncInfo.CIPHER, None, 24, None, block_size=16, key_size=16, iv_size=16, year=2013, ctype=FConstType.ARX),
    FuncInfo('LED', FuncInfo.CIPHER, None, 48, None, block_size=8, key_size=10, iv_size=16, year=2010, ctype=FConstType.SPN),
    FuncInfo('MARS', FuncInfo.CIPHER, None, 16, None, block_size=16, key_size=16, iv_size=16, year=1999, ctype=FConstType.FN),
    FuncInfo('MISTY1', FuncInfo.CIPHER, None, 8, 6, block_size=8, key_size=16, iv_size=16, year=1995, ctype=FConstType.FN),
    FuncInfo('NOEKEON', FuncInfo.CIPHER, None, 16, 4, block_size=16, key_size=16, iv_size=16, year=2009, ctype=FConstType.SPN),  # https://en.wikipedia.org/wiki/NOEKEON
    FuncInfo('PICCOLO', FuncInfo.CIPHER, None, 25, None, block_size=8, key_size=10, iv_size=16, year=2011, ctype=FConstType.FN),
    FuncInfo('PRIDE', FuncInfo.CIPHER, None, 20, 19, block_size=8, key_size=16, iv_size=16, year=2014, ctype=FConstType.SPN),  # https://eprint.iacr.org/2014/656.pdf
    FuncInfo('Prince', FuncInfo.CIPHER, None, 12, 6, block_size=8, key_size=16, iv_size=16, year=2012, ctype=FConstType.SPN),  # https://en.wikipedia.org/wiki/Prince_(cipher)
    FuncInfo('RC5-20', FuncInfo.CIPHER, None, 20, None, block_size=8, key_size=16, iv_size=16, year=1997, ctype=FConstType.FN),
    FuncInfo('RC6', FuncInfo.CIPHER, None, 20, None, block_size=16, key_size=16, iv_size=16, year=1997, ctype=FConstType.GFN),
    FuncInfo('RECTANGLE-K128', FuncInfo.CIPHER, FuncInfo.LIGHT, 25, 14, block_size=8, key_size=16, iv_size=16, year=2014, ctype=FConstType.SPN),  # https://eprint.iacr.org/2014/084.pdf
    FuncInfo('RECTANGLE-K80', FuncInfo.CIPHER, FuncInfo.LIGHT, 25, 18, block_size=8, key_size=10, iv_size=16, year=2014, ctype=FConstType.SPN),
    FuncInfo('ROAD-RUNNER-K128', FuncInfo.CIPHER, FuncInfo.LIGHT, 12, None, block_size=8, key_size=16, iv_size=16, year=2015, ctype=FConstType.FN),  # https://eprint.iacr.org/2015/906.pdf
    FuncInfo('ROAD-RUNNER-K80', FuncInfo.CIPHER, FuncInfo.LIGHT, 10, None, block_size=8, key_size=10, iv_size=16, year=2015, ctype=FConstType.FN),
    FuncInfo('ROBIN', FuncInfo.CIPHER, None, 16, 16, block_size=16, key_size=16, iv_size=16, year=2014, ctype=FConstType.SPN),  # https://who.paris.inria.fr/Gaetan.Leurent/files/LS_FSE14.pdf
    FuncInfo('ROBIN-STAR', FuncInfo.CIPHER, None, 16, None, block_size=16, key_size=16, iv_size=16, year=2017, ctype=FConstType.SPN),  # https://fenix.tecnico.ulisboa.pt/downloadFile/281870113704550/Extended_Abstract-Choosing_the_Future_of_Lightweight_Encryption_Algorithms.pdf
    FuncInfo('SEED', FuncInfo.CIPHER, None, 16, None, block_size=16, key_size=16, iv_size=16, year=1998, ctype=FConstType.FN),  # https://en.wikipedia.org/wiki/SEED
    FuncInfo('SERPENT', FuncInfo.CIPHER, None, 32, 5, block_size=16, key_size=16, iv_size=16, year=1998, ctype=FConstType.SPN),  # https://en.wikipedia.org/wiki/Serpent_(cipher)
    FuncInfo('SHACAL2', FuncInfo.CIPHER, None, 80, 44, block_size=32, key_size=64, iv_size=16, year=2000, ctype=FConstType.HASH),  # https://www.researchgate.net/publication/220237385_Attacking_44_Rounds_of_the_SHACAL-2_Block_Cipher_Using_Related-Key_Rectangle_Cryptanalysis https://github.com/odzhan/tinycrypt/blob/master/block/shacal2/doc/10.1.1.3.4066.pdf
    FuncInfo('SIMON', FuncInfo.CIPHER, None, 68, 17, block_size=16, key_size=16, iv_size=16, year=2013, ctype=FConstType.SPN),
    FuncInfo('SINGLE-DES', FuncInfo.CIPHER, None, 16, 16, block_size=8, key_size=7, iv_size=16, year=1977, ctype=FConstType.FN),  # https://en.wikipedia.org/wiki/Data_Encryption_Standard
    FuncInfo('SPARX-B128', FuncInfo.CIPHER, FuncInfo.LIGHT, 32, 8, block_size=16, key_size=16, iv_size=16, year=2016, ctype=FConstType.ARX),  # https://www.cryptolux.org/index.php/SPARX
    FuncInfo('SPARX-B64', FuncInfo.CIPHER, FuncInfo.LIGHT, 24, 8, block_size=8, key_size=16, iv_size=16, year=2016, ctype=FConstType.ARX),
    FuncInfo('SPECK', FuncInfo.CIPHER, FuncInfo.LIGHT, 32, 15, block_size=16, key_size=16, iv_size=16, year=2013, ctype=FConstType.ARX),
    FuncInfo('TEA', FuncInfo.CIPHER, FuncInfo.LIGHT, 32, 5, block_size=8, key_size=16, iv_size=16, year=1994, ctype=FConstType.FN),
    FuncInfo('TRIPLE-DES', FuncInfo.CIPHER, None, 16, None, block_size=8, key_size=21, iv_size=16, year=1981, ctype=FConstType.FN),  # https://en.wikipedia.org/wiki/Triple_DES
    FuncInfo('TWINE', FuncInfo.CIPHER, FuncInfo.LIGHT, 35, 23, block_size=8, key_size=10, iv_size=16, year=2011, ctype=FConstType.FN),  # https://www.nec.com/en/global/rd/tg/code/symenc/pdf/twine_LC11.pdf
    FuncInfo('TWOFISH', FuncInfo.CIPHER, None, 16, 16, block_size=16, key_size=16, iv_size=16, year=1998, ctype=FConstType.FN),  # https://en.wikipedia.org/wiki/Twofish
    FuncInfo('XTEA', FuncInfo.CIPHER, FuncInfo.LIGHT, 32, 8, block_size=8, key_size=16, iv_size=16, year=2017, ctype=FConstType.FN),

    FuncInfo('ARIRANG', FuncInfo.HASH, None, 4, 4, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/ARIRANG
    FuncInfo('AURORA', FuncInfo.HASH, None, 17, None, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/AURORA
    FuncInfo('Abacus', FuncInfo.HASH, None, 280, None, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/Abacus
    FuncInfo('BLAKE', FuncInfo.HASH, None, 14, 4, block_size=32, year=2008, ctype=FConstType.ARX),  # https://ehash.iaik.tugraz.at/wiki/BLAKE
    FuncInfo('BMW', FuncInfo.HASH, None, 16, None, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/Blue_Midnight_Wish
    FuncInfo('Blender', FuncInfo.HASH, None, 32, None, block_size=64, year=2008),  # https://ehash.iaik.tugraz.at/wiki/Blender
    FuncInfo('Boole', FuncInfo.HASH, None, 16, 16, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/Boole
    FuncInfo('CHI', FuncInfo.HASH, None, 20, None, block_size=64, year=2008),  # https://ehash.iaik.tugraz.at/wiki/CHI
    FuncInfo('Cheetah', FuncInfo.HASH, None, 16, 12, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/Cheetah
    FuncInfo('CubeHash', FuncInfo.HASH, None, 8, None, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/CubeHash
    FuncInfo('DCH', FuncInfo.HASH, None, 4, 4, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/DCH
    FuncInfo('DynamicSHA', FuncInfo.HASH, None, 16, None, block_size=48, hash_size=64, year=2008),  # https://ehash.iaik.tugraz.at/wiki/Dynamic_SHA
    FuncInfo('DynamicSHA2', FuncInfo.HASH, None, 17, 17, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/Dynamic_SHA2
    FuncInfo('ECHO', FuncInfo.HASH, None, 8, 4, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/ECHO
    FuncInfo('ESSENCE', FuncInfo.HASH, None, 32, None, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/ESSENCE
    FuncInfo('Gost', FuncInfo.HASH, None, 32, 5, block_size=32, year=1994),  # https://en.wikipedia.org/wiki/GOST_(hash_function)
    FuncInfo('Grostl', FuncInfo.HASH, None, 10, None, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/Groestl
    FuncInfo('Hamsi', FuncInfo.HASH, None, 3, None, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/Hamsi
    FuncInfo('JH', FuncInfo.HASH, None, 42, 10, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/JH
    FuncInfo('Keccak', FuncInfo.HASH, None, 24, 5, block_size=32, year=2008, ctype=FConstType.SPONGE),  # https://ehash.iaik.tugraz.at/wiki/Keccak
    FuncInfo('Lesamnta', FuncInfo.HASH, None, 32, 32, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/Lesamnta
    FuncInfo('Luffa', FuncInfo.HASH, None, 8, 8, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/Luffa
    FuncInfo('MCSSHA3', FuncInfo.HASH, None, 1, None, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/MCSSHA-3
    FuncInfo('MD5', FuncInfo.HASH, None, 64, 64, block_size=16, year=1992, ctype=FConstType.MERKLED),  # https://ehash.iaik.tugraz.at/wiki/MD5
    FuncInfo('MD6', FuncInfo.HASH, None, 104, 16, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/MD6
    FuncInfo('RIPEMD160', FuncInfo.HASH, None, 80, 48, block_size=20, year=1992),  # https://ehash.iaik.tugraz.at/wiki/RIPEMD  https://en.wikipedia.org/wiki/RIPEMD
    FuncInfo('SHA1', FuncInfo.HASH, None, 80, 80, block_size=20, year=1995, ctype=FConstType.MERKLED),  # https://en.wikipedia.org/wiki/SHA-1
    FuncInfo('SHA2', FuncInfo.HASH, None, 64, 31, block_size=32, year=2001, ctype=FConstType.MERKLED),  # https://en.wikipedia.org/wiki/SHA-2
    FuncInfo('SHA3', FuncInfo.HASH, None, 24, 5, block_size=32, year=2016, ctype=FConstType.SPONGE),  # https://en.wikipedia.org/wiki/SHA-3
    FuncInfo('SHAvite3', FuncInfo.HASH, None, 12, None, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/SHAvite-3
    FuncInfo('SIMD', FuncInfo.HASH, None, 4, None, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/SIMD
    FuncInfo('Sarmal', FuncInfo.HASH, None, 16, None, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/Sarmal
    FuncInfo('Shabal', FuncInfo.HASH, None, 1, None, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/Shabal
    FuncInfo('Skein', FuncInfo.HASH, None, 72, 17, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/Skein
    FuncInfo('TIB3', FuncInfo.HASH, None, 16, None, block_size=48, year=2008),  # https://ehash.iaik.tugraz.at/wiki/TIB3
    FuncInfo('Tangle', FuncInfo.HASH, None, 80, 80, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/Tangle  https://ehash.iaik.tugraz.at/uploads/4/40/Tangle.pdf
    FuncInfo('Tangle2', FuncInfo.HASH, None, 80, None, block_size=32, year=2008),
    FuncInfo('Tiger', FuncInfo.HASH, None, 23, 19, block_size=24, year=1996, ctype=FConstType.MERKLED),  # https://ehash.iaik.tugraz.at/wiki/Tiger
    FuncInfo('Twister', FuncInfo.HASH, None, 9, 9, block_size=32, year=2008),  # https://ehash.iaik.tugraz.at/wiki/Twister
    FuncInfo('Whirlpool', FuncInfo.HASH, None, 10, 6, block_size=64, year=2000, ctype=FConstType.MPRENEL),  # https://ehash.iaik.tugraz.at/wiki/Whirlpool  https://en.wikipedia.org/wiki/Whirlpool_(hash_function)

    FuncInfo('Chacha', FuncInfo.CIPHER, FuncInfo.STREAM, 20, None, block_size=32, key_size=32, iv_size=8, year=2008, ctype=FConstType.ARX),  # https://cr.yp.to/chacha.html
    FuncInfo('DECIM', FuncInfo.CIPHER, FuncInfo.STREAM, 8, None, block_size=24, key_size=10, iv_size=8, year=2005),  # https://en.wikipedia.org/wiki/DECIM
    FuncInfo('F-FCSR', FuncInfo.CIPHER, FuncInfo.STREAM, 5, 5, block_size=16, key_size=16, iv_size=8, year=2005),  # https://en.wikipedia.org/wiki/F-FCSR  https://www.iacr.org/archive/asiacrypt2008/53500563/53500563.pdf
    FuncInfo('Fubuki', FuncInfo.CIPHER, FuncInfo.STREAM, 4, None, block_size=16, key_size=16, iv_size=16, year=2005),  # https://www.ecrypt.eu.org/stream/cryptmtfubuki.html
    FuncInfo('Grain', FuncInfo.CIPHER, FuncInfo.STREAM, 13, None, block_size=16, key_size=16, iv_size=12, year=2005),  # https://www.ecrypt.eu.org/stream/grain.html
    FuncInfo('HC-128', FuncInfo.CIPHER, FuncInfo.STREAM, 1, None, block_size=16, key_size=16, iv_size=16, year=2005),  # https://www.ecrypt.eu.org/stream/hc256.html
    FuncInfo('Hermes', FuncInfo.CIPHER, FuncInfo.STREAM, 10, None, block_size=16, key_size=10, iv_size=0, year=2005),  # https://www.ecrypt.eu.org/stream/hermes8.html
    FuncInfo('LEX', FuncInfo.CIPHER, FuncInfo.STREAM, 10, None, block_size=16, key_size=16, iv_size=16, year=2005),  # https://www.ecrypt.eu.org/stream/lex.html
    FuncInfo('MICKEY', FuncInfo.CIPHER, FuncInfo.STREAM, 1, None, block_size=16, key_size=16, iv_size=0, year=2005),
    FuncInfo('RC4', FuncInfo.CIPHER, FuncInfo.STREAM, 1, None, block_size=32, key_size=16, iv_size=0, year=1994),  # https://en.wikipedia.org/wiki/RC4
    FuncInfo('Rabbit', FuncInfo.CIPHER, FuncInfo.STREAM, 4, None, block_size=16, key_size=16, iv_size=8, year=2005),  # https://en.wikipedia.org/wiki/Rabbit_(cipher)
    FuncInfo('SOSEMANUK', FuncInfo.CIPHER, FuncInfo.STREAM, 25, None, block_size=16, key_size=16, iv_size=16, year=2005),  # https://en.wikipedia.org/wiki/SOSEMANUK
    FuncInfo('Salsa20', FuncInfo.CIPHER, FuncInfo.STREAM, 20, None, block_size=8, key_size=16, iv_size=8, year=2007, ctype=FConstType.ARX),  # https://en.wikipedia.org/wiki/Salsa20
    FuncInfo('TSC-4', FuncInfo.CIPHER, FuncInfo.STREAM, 32, None, block_size=32, key_size=10, iv_size=10, year=2006),  # https://www.ecrypt.eu.org/stream/papersdir/2006/024.pdf
    FuncInfo('Trivium', FuncInfo.CIPHER, FuncInfo.STREAM, 9, None, block_size=8, key_size=10, iv_size=10, year=2005),  # https://en.wikipedia.org/wiki/Trivium_(cipher)

    FuncInfo('gmimc-S45a', FuncInfo.CIPHER, FuncInfo.MPC, 121, year=2019, ctype=FConstType.FN),  # https://eprint.iacr.org/2019/397
    FuncInfo('gmimc-S45b', FuncInfo.CIPHER, FuncInfo.MPC, 137, year=2019, ctype=FConstType.FN),
    FuncInfo('gmimc-S80a', FuncInfo.CIPHER, FuncInfo.MPC, 111, year=2019, ctype=FConstType.FN),
    FuncInfo('gmimc-S80b', FuncInfo.CIPHER, FuncInfo.MPC, 210, year=2019, ctype=FConstType.FN),
    FuncInfo('gmimc-S80c', FuncInfo.CIPHER, FuncInfo.MPC, 226, year=2019, ctype=FConstType.FN),
    FuncInfo('gmimc-S128a', FuncInfo.CIPHER, FuncInfo.MPC, 166, year=2019, ctype=FConstType.FN),
    FuncInfo('gmimc-S128b', FuncInfo.CIPHER, FuncInfo.MPC, 166, year=2019, ctype=FConstType.FN),
    FuncInfo('gmimc-S128c', FuncInfo.CIPHER, FuncInfo.MPC, 182, year=2019, ctype=FConstType.FN),
    FuncInfo('gmimc-S128d', FuncInfo.CIPHER, FuncInfo.MPC, 101, year=2019, ctype=FConstType.FN),
    FuncInfo('gmimc-S128e', FuncInfo.CIPHER, FuncInfo.MPC, 342, year=2019, ctype=FConstType.FN),
    FuncInfo('gmimc-S256f', FuncInfo.CIPHER, FuncInfo.MPC, 174, year=2019, ctype=FConstType.FN),
    FuncInfo('gmimc-S256b', FuncInfo.CIPHER, FuncInfo.MPC, 186, year=2019, ctype=FConstType.FN),

    FuncInfo('mimc_hash-S45', FuncInfo.CIPHER, FuncInfo.MPC, 116, year=2016, ctype=FConstType.SPN),  # https://mimc.iaik.tugraz.at/pages/mimc.php
    FuncInfo('mimc_hash-S80', FuncInfo.CIPHER, FuncInfo.MPC, 204, year=2016, ctype=FConstType.SPN),
    FuncInfo('mimc_hash-S128', FuncInfo.CIPHER, FuncInfo.MPC, 320, year=2016, ctype=FConstType.SPN),

    FuncInfo('lowmc-s80a', FuncInfo.CIPHER, FuncInfo.MPC, 12, year=2020, ctype=FConstType.SPN),
    FuncInfo('lowmc-s80b', FuncInfo.CIPHER, FuncInfo.MPC, 12, year=2020, ctype=FConstType.SPN),
    FuncInfo('lowmc-s128a', FuncInfo.CIPHER, FuncInfo.MPC, 14, year=2020, ctype=FConstType.SPN),
    FuncInfo('lowmc-s128b', FuncInfo.CIPHER, FuncInfo.MPC, 252, year=2020, ctype=FConstType.SPN),
    FuncInfo('lowmc-s128c', FuncInfo.CIPHER, FuncInfo.MPC, 128, year=2020, ctype=FConstType.SPN),
    FuncInfo('lowmc-s128d', FuncInfo.CIPHER, FuncInfo.MPC, 88, year=2020, ctype=FConstType.SPN),

    FuncInfo('Poseidon_S45a', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=26, year=2019),  # https://eprint.iacr.org/2019/458
    FuncInfo('Starkad_S45a', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=28, year=2019),
    FuncInfo('Poseidon_S45b', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=28, year=2019),
    FuncInfo('Starkad_S45b', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=31, year=2019),
    FuncInfo('Poseidon_S80a', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=51, year=2019),
    FuncInfo('Starkad_S80a', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=53, year=2019),
    FuncInfo('Poseidon_S80b', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=50, year=2019),
    FuncInfo('Starkad_S80b', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=52, year=2019),
    FuncInfo('Poseidon_S80c', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=52, year=2019),
    FuncInfo('Starkad_S80c', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=54, year=2019),
    FuncInfo('Poseidon_S128a', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=81, year=2019),
    FuncInfo('Starkad_S128a', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=85, year=2019),
    FuncInfo('Poseidon_S128b', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=83, year=2019),
    FuncInfo('Starkad_S128b', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=85, year=2019),
    FuncInfo('Poseidon_S128c', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=83, year=2019),
    FuncInfo('Starkad_S128c', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=86, year=2019),
    FuncInfo('Poseidon_S128d', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=40, year=2019),
    FuncInfo('Starkad_S128d', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=43, year=2019),
    FuncInfo('Poseidon_S128e', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=85, year=2019),
    FuncInfo('Starkad_S128e', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=88, year=2019),
    FuncInfo('Poseidon_S256a', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=82, year=2019),
    FuncInfo('Starkad_S256a', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=86, year=2019),
    FuncInfo('Poseidon_S256b', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=83, year=2019),
    FuncInfo('Starkad_S256b', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=86, year=2019),
    FuncInfo('Poseidon_S128_BLS12_138', FuncInfo.CIPHER, FuncInfo.MPC, 8, None, partial_rounds=60, year=2019),

    FuncInfo('Rescue_S45a', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),  # https://eprint.iacr.org/2019/426
    FuncInfo('Vision_S45a', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),
    FuncInfo('Rescue_S45b', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),
    FuncInfo('Vision_S45b', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),
    FuncInfo('Rescue_S80a', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),
    FuncInfo('Vision_S80a', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),
    FuncInfo('Rescue_S80b', FuncInfo.CIPHER, FuncInfo.MPC, 14, None, year=2019),
    FuncInfo('Vision_S80b', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),
    FuncInfo('Rescue_S80c', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),
    FuncInfo('Vision_S80c', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),
    FuncInfo('Rescue_S128a', FuncInfo.CIPHER, FuncInfo.MPC, 16, None, year=2019),
    FuncInfo('Vision_S128a', FuncInfo.CIPHER, FuncInfo.MPC, 12, None, year=2019),
    FuncInfo('Rescue_S128b', FuncInfo.CIPHER, FuncInfo.MPC, 22, None, year=2019),
    FuncInfo('Vision_S128b', FuncInfo.CIPHER, FuncInfo.MPC, 16, None, year=2019),
    FuncInfo('Rescue_S128c', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),
    FuncInfo('Vision_S128c', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),
    FuncInfo('Rescue_S128d', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),
    FuncInfo('Vision_S128d', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),
    FuncInfo('Rescue_S128e', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),
    FuncInfo('Vision_S128e', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),
    FuncInfo('Rescue_S256a', FuncInfo.CIPHER, FuncInfo.MPC, 16, None, year=2019),
    FuncInfo('Vision_S256a', FuncInfo.CIPHER, FuncInfo.MPC, 12, None, year=2019),
    FuncInfo('Rescue_S256b', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),
    FuncInfo('Vision_S256b', FuncInfo.CIPHER, FuncInfo.MPC, 10, None, year=2019),

    FuncInfo('RescueP_S80a', FuncInfo.CIPHER, FuncInfo.MPC, 18, None, year=2020),  # https://www.esat.kuleuven.be/cosic/sites/rescue/
    FuncInfo('RescueP_S80b', FuncInfo.CIPHER, FuncInfo.MPC, 18, None, year=2020),
    FuncInfo('RescueP_S80c', FuncInfo.CIPHER, FuncInfo.MPC, 9, None, year=2020),
    FuncInfo('RescueP_S80d', FuncInfo.CIPHER, FuncInfo.MPC, 9, None, year=2020),
    FuncInfo('RescueP_128a', FuncInfo.CIPHER, FuncInfo.MPC, 27, None, year=2020),
    FuncInfo('RescueP_128b', FuncInfo.CIPHER, FuncInfo.MPC, 27, None, year=2020),
    FuncInfo('RescueP_128c', FuncInfo.CIPHER, FuncInfo.MPC, 14, None, year=2020),
    FuncInfo('RescueP_128d', FuncInfo.CIPHER, FuncInfo.MPC, 14, None, year=2020),

    FuncInfo('std_mersenne_twister', FuncInfo.PRNG, max_rounds=0, key_size=4, block_size=4, year=1997),  # https://en.wikipedia.org/wiki/Mersenne_Twister
    FuncInfo('std_lcg', FuncInfo.PRNG, max_rounds=0, key_size=4, block_size=4),  # https://en.wikipedia.org/wiki/Linear_congruential_generator
    FuncInfo('std_subtract_with_carry', FuncInfo.PRNG, max_rounds=0, key_size=4, block_size=4),
    FuncInfo('testu01-ulcg', FuncInfo.PRNG, max_rounds=0, key_size=6, block_size=7),
    FuncInfo('testu01-umrg', FuncInfo.PRNG, max_rounds=0, key_size=6, block_size=7),
    FuncInfo('testu01-uxorshift', FuncInfo.PRNG, max_rounds=0, key_size=32, block_size=4),

])
