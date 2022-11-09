# Jobs gen

```python
import os
from rtt_tools import generator_mpc as g
os.chdir('/tmp/ggen7')
rr = g.generate_stream_col('RC4', 100*1024*1024, 1, 32, 16, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('RC4', 10*1024*1024, 1, 32, 16, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('MICKEY', 100*1024*1024, 1, 16, 16, eprefix='PH4-SM-07-')
g.write_submit_obj(rr)
```


```python
import os, shutil
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen8'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname)
rr = g.generate_block_col('AES', 100*1024*1024, 3, 16, 16, eprefix='PH4-SM-07-')
rr += g.generate_block_col('AES', 100*1024*1024, 4, 16, 16, eprefix='PH4-SM-07-')
rr += g.generate_block_col('AES', 100*1024*1024, 5, 16, 16, eprefix='PH4-SM-07-')
rr += g.generate_block_col('AES', 1000*1024*1024, 4, 16, 16, eprefix='PH4-SM-07-')
rr += g.generate_block_col('AES', 1000*1024*1024, 5, 16, 16, eprefix='PH4-SM-07-')
g.write_submit_obj(rr)
```

```python
import os, shutil
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen9'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname)
rr = g.generate_block_col('AES', 100*1024*1024, 1, 16, 16, eprefix='PH4-SM-07-')
rr += g.generate_block_col('AES', 100*1024*1024, 2, 16, 16, eprefix='PH4-SM-07-')
g.write_submit_obj(rr)
```

## CTR LHW for stream:col
```python
import os, shutil
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen10'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]
rr += g.generate_stream_col('Chacha', 100*1024*1024, 2, 32, 32, 8, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('Chacha', 100*1024*1024, 3, 32, 32, 8, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('Chacha', 100*1024*1024, 4, 32, 32, 8, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('Chacha', 100*1024*1024, 5, 32, 32, 8, eprefix='PH4-SM-07-')

rr += g.generate_stream_col('DECIM', 100*1024*1024, 6, 24, 10, 8, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('DECIM', 100*1024*1024, 7, 24, 10, 8, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('DECIM', 100*1024*1024, 8, 24, 10, 8, eprefix='PH4-SM-07-')

rr += g.generate_stream_col('F-FCSR', 100*1024*1024, 1, 16, 16, 8, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('F-FCSR', 100*1024*1024, 2, 16, 16, 8, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('F-FCSR', 100*1024*1024, 3, 16, 16, 8, eprefix='PH4-SM-07-')

rr += g.generate_stream_col('Fubuki', 100*1024*1024, 1, 16, 16, 16, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('Fubuki', 100*1024*1024, 2, 16, 16, 16, eprefix='PH4-SM-07-')

rr += g.generate_stream_col('Grain', 100*1024*1024, 2, 16, 16, 12, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('Grain', 100*1024*1024, 3, 16, 16, 12, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('Grain', 100*1024*1024, 4, 16, 16, 12, eprefix='PH4-SM-07-')

rr += g.generate_stream_col('HC-128', 100*1024*1024, 1, 16, 16, 16, eprefix='PH4-SM-07-')

rr += g.generate_stream_col('Hermes', 100*1024*1024, 1, 16, 10, 0, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('Hermes', 100*1024*1024, 2, 16, 10, 0, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('Hermes', 100*1024*1024, 3, 16, 10, 0, eprefix='PH4-SM-07-')

rr += g.generate_stream_col('LEX', 100*1024*1024, 3, 16, 16, 16, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('LEX', 100*1024*1024, 4, 16, 16, 16, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('LEX', 100*1024*1024, 5, 16, 16, 16, eprefix='PH4-SM-07-')

rr += g.generate_stream_col('Rabbit', 100*1024*1024, 1, 16, 16, 8, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('Rabbit', 100*1024*1024, 2, 16, 16, 8, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('Rabbit', 100*1024*1024, 3, 16, 16, 8, eprefix='PH4-SM-07-')

rr += g.generate_stream_col('Salsa20', 100*1024*1024, 1, 8, 16, 8, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('Salsa20', 100*1024*1024, 2, 8, 16, 8, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('Salsa20', 100*1024*1024, 3, 8, 16, 8, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('Salsa20', 100*1024*1024, 4, 8, 16, 8, eprefix='PH4-SM-07-')

rr += g.generate_stream_col('SOSEMANUK', 100*1024*1024, 4, 16, 16, 16, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('SOSEMANUK', 100*1024*1024, 5, 16, 16, 16, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('SOSEMANUK', 100*1024*1024, 6, 16, 16, 16, eprefix='PH4-SM-07-')

rr += g.generate_stream_col('Trivium', 100*1024*1024, 1, 8, 10, 10, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('Trivium', 100*1024*1024, 2, 8, 10, 10, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('Trivium', 100*1024*1024, 3, 8, 10, 10, eprefix='PH4-SM-07-')

rr += g.generate_stream_col('TSC-4', 100*1024*1024, 14, 32, 10, 10, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('TSC-4', 100*1024*1024, 15, 32, 10, 10, eprefix='PH4-SM-07-')
rr += g.generate_stream_col('TSC-4', 100*1024*1024, 16, 32, 10, 10, eprefix='PH4-SM-07-')

g.write_submit_obj(rr)
```


## SAC RND for stream:col

```python
import os, shutil
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen11'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]
rr += g.generate_stream_col('Chacha', 100*1024*1024, 2, 32, 32, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('Chacha', 100*1024*1024, 3, 32, 32, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('Chacha', 100*1024*1024, 4, 32, 32, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('Chacha', 100*1024*1024, 5, 32, 32, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)

rr += g.generate_stream_col('DECIM', 100*1024*1024, 6, 24, 10, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('DECIM', 100*1024*1024, 7, 24, 10, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('DECIM', 100*1024*1024, 8, 24, 10, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)

rr += g.generate_stream_col('F-FCSR', 100*1024*1024, 1, 16, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('F-FCSR', 100*1024*1024, 2, 16, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('F-FCSR', 100*1024*1024, 3, 16, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)

rr += g.generate_stream_col('Fubuki', 100*1024*1024, 1, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('Fubuki', 100*1024*1024, 2, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)

rr += g.generate_stream_col('Grain', 100*1024*1024, 2, 16, 16, 12, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('Grain', 100*1024*1024, 3, 16, 16, 12, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('Grain', 100*1024*1024, 4, 16, 16, 12, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)

rr += g.generate_stream_col('HC-128', 100*1024*1024, 1, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)

rr += g.generate_stream_col('Hermes', 100*1024*1024, 1, 16, 10, 0, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('Hermes', 100*1024*1024, 2, 16, 10, 0, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('Hermes', 100*1024*1024, 3, 16, 10, 0, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)

rr += g.generate_stream_col('LEX', 100*1024*1024, 3, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('LEX', 100*1024*1024, 4, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('LEX', 100*1024*1024, 5, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)

rr += g.generate_stream_col('Rabbit', 100*1024*1024, 1, 16, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('Rabbit', 100*1024*1024, 2, 16, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('Rabbit', 100*1024*1024, 3, 16, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)

rr += g.generate_stream_col('Salsa20', 100*1024*1024, 1, 8, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('Salsa20', 100*1024*1024, 2, 8, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('Salsa20', 100*1024*1024, 3, 8, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('Salsa20', 100*1024*1024, 4, 8, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)

rr += g.generate_stream_col('SOSEMANUK', 100*1024*1024, 4, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('SOSEMANUK', 100*1024*1024, 5, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('SOSEMANUK', 100*1024*1024, 6, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('SOSEMANUK', 100*1024*1024, 7, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('SOSEMANUK', 100*1024*1024, 8, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('SOSEMANUK', 100*1024*1024, 9, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('Trivium', 100*1024*1024, 1, 8, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('Trivium', 100*1024*1024, 2, 8, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('Trivium', 100*1024*1024, 3, 8, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('Trivium', 100*1024*1024, 4, 8, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Trivium', 100*1024*1024, 5, 8, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Trivium', 100*1024*1024, 6, 8, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Trivium', 100*1024*1024, 7, 8, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('TSC-4', 100*1024*1024, 14, 32, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('TSC-4', 100*1024*1024, 15, 32, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
rr += g.generate_stream_col('TSC-4', 100*1024*1024, 16, 32, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)

g.write_submit_obj(rr)
```

## CTR LHW SAC RND for stream:col 1000MB for bound rounds
```python
import os, shutil
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen12'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]
rr += g.generate_stream_col('Chacha', 1000*1024*1024, 4, 32, 32, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Chacha', 1000*1024*1024, 5, 32, 32, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('DECIM', 1000*1024*1024, 6, 24, 10, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('DECIM', 1000*1024*1024, 7, 24, 10, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('F-FCSR', 1000*1024*1024, 4, 16, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('F-FCSR', 1000*1024*1024, 5, 16, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('Fubuki', 1000*1024*1024, 1, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Fubuki', 1000*1024*1024, 2, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('Grain', 1000*1024*1024, 4, 16, 16, 12, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Grain', 1000*1024*1024, 5, 16, 16, 12, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Grain', 1000*1024*1024, 6, 16, 16, 12, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('HC-128', 1000*1024*1024, 1, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('Hermes', 1000*1024*1024, 1, 16, 10, 0, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Hermes', 1000*1024*1024, 2, 16, 10, 0, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Hermes', 1000*1024*1024, 3, 16, 10, 0, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('LEX', 1000*1024*1024, 4, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('Rabbit', 1000*1024*1024, 1, 16, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('RC4', 1000*1024*1024, 1, 32, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('Salsa20', 1000*1024*1024, 3, 8, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Salsa20', 1000*1024*1024, 4, 8, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('SOSEMANUK', 1000*1024*1024, 7, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('SOSEMANUK', 1000*1024*1024, 8, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('Trivium', 1000*1024*1024, 4, 8, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Trivium', 1000*1024*1024, 5, 8, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('TSC-4', 1000*1024*1024, 14, 32, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('TSC-4', 1000*1024*1024, 15, 32, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

g.write_submit_obj(rr)
```

# Fixing redundant 1GB stream ciphers

```python
import os, shutil
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen13'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]
rr += g.generate_stream_col('DECIM', 1000*1024*1024, 6, 24, 10, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)
rr += g.generate_stream_col('DECIM', 1000*1024*1024, 7, 24, 10, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)

rr += g.generate_stream_col('Hermes', 1000*1024*1024, 1, 16, 10, 0, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)
rr += g.generate_stream_col('Hermes', 1000*1024*1024, 2, 16, 10, 0, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)
rr += g.generate_stream_col('Hermes', 1000*1024*1024, 3, 16, 10, 0, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)

rr += g.generate_stream_col('Salsa20', 1000*1024*1024, 3, 8, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)
rr += g.generate_stream_col('Salsa20', 1000*1024*1024, 4, 8, 16, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)

rr += g.generate_stream_col('TSC-4', 1000*1024*1024, 14, 32, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)
rr += g.generate_stream_col('TSC-4', 1000*1024*1024, 15, 32, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)

g.write_submit_obj(rr)
```

# Redundant for 100MB
```python
import os, shutil
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen15'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]
rr += g.generate_stream_col('RC4', 100*1024*1024, 1, 32, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)

rr += g.generate_stream_col('DECIM', 100*1024*1024, 6, 24, 10, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)
rr += g.generate_stream_col('DECIM', 100*1024*1024, 7, 24, 10, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)
rr += g.generate_stream_col('DECIM', 100*1024*1024, 8, 24, 10, 8, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)

rr += g.generate_stream_col('Hermes', 100*1024*1024, 1, 16, 10, 0, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)
rr += g.generate_stream_col('Hermes', 100*1024*1024, 2, 16, 10, 0, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)
rr += g.generate_stream_col('Hermes', 100*1024*1024, 3, 16, 10, 0, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)

rr += g.generate_stream_col('TSC-4', 100*1024*1024, 14, 32, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)
rr += g.generate_stream_col('TSC-4', 100*1024*1024, 15, 32, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)
rr += g.generate_stream_col('TSC-4', 100*1024*1024, 16, 32, 10, 10, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)

g.write_submit_obj(rr)
```

# RC4
```python
import os, shutil
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen19'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]
rr += g.generate_stream_col('RC4', 100*1024*1024, 1, 32, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.SAC_RND)
g.write_submit_obj(rr)
```

# Next new rounds
```python
import os, shutil
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen20'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]
rr += g.generate_stream_col('RC4', 100*1024*1024, 1, 32, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.LHW)

rr += g.generate_stream_col('Grain', 100*1024*1024, 8, 16, 16, 12, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Grain', 100*1024*1024, 9, 16, 16, 12, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Grain', 100*1024*1024, 10, 16, 16, 12, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Grain', 100*1024*1024, 11, 16, 16, 12, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Grain', 100*1024*1024, 12, 16, 16, 12, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Grain', 1000*1024*1024, 7, 16, 16, 12, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('SOSEMANUK', 100*1024*1024, 10, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('SOSEMANUK', 100*1024*1024, 11, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('SOSEMANUK', 100*1024*1024, 12, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('SOSEMANUK', 1000*1024*1024, 9, 16, 16, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

g.write_submit_obj(rr)
```

# LowMC manual

```python
import os, shutil
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen21'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]

to_gen = [
    ('lowmc-s128b', (252, ), [12, 13, 14, 15, 16]),
    ('lowmc-s128c', (128, ), [12, 13, 14, 15, 16]),
    ('lowmc-s128d', (88, ), [12, 13, 14, 15, 16, 51, 52, 53, 147, 148, 149]),
]

rr += g.gen_lowmc_core(to_gen, [10 * 1024 * 1024, 100 * 1024 * 1024], eprefix='testmpc21-')
g.write_submit_obj(rr)
```

# LowMC manual missing

```python
import os, shutil
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen22'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]

to_gen = [
    ('lowmc-s80a', (None, ), [1, 2, 5]),
    ('lowmc-s80b', (None, ), [1, 2, 5]),
    ('lowmc-s128a', (None, ), [1, 2, 5]),
    ('lowmc-s128b', (None, ), [1, 2, 5]),
    ('lowmc-s128c', (None, ), [1, 2, 5]),
    ('lowmc-s128d', (None, ), [1, 2, 5]),
]

rr += g.gen_lowmc_core(to_gen, [10 * 1024 * 1024, 100 * 1024 * 1024], eprefix='testmpc22-')
g.write_submit_obj(rr)
```

# Single DES weak keys
```python
import os, shutil, itertools
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen23'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]
rr += list(itertools.chain.from_iterable(
    [g.generate_block_col('SINGLE-DES', 100*1024*1024, r, 8, 7, 16, eprefix='PH4-SM-07-', streams=g.StreamOptions.CTR_LHW_SAC_RND) for r in range(1, 17)]))
g.write_submit_obj(rr)
```

# LowMC manual 2

```python
import os, shutil
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen24'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]

to_gen = [
    ('lowmc-s128b', (252, ), [17, 18, 19, 21, 89, 91]),
]

rr += g.gen_lowmc_core(to_gen, [10 * 1024 * 1024, 100 * 1024 * 1024], eprefix='testmpc21-')
g.write_submit_obj(rr)
```


# Next new rounds
```python
import os, shutil, itertools
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen25'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]
rr += g.generate_stream_col('RC4', 10*1024*1024, 1, 32, 16, eprefix='PH4-SM-25-', streams=g.StreamOptions.SAC_RND)

rr += g.generate_stream_col('Grain', 1000*1024*1024, 9, 16, 16, 12, eprefix='PH4-SM-25-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_stream_col('Grain', 1000*1024*1024, 10, 16, 16, 12, eprefix='PH4-SM-25-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += g.generate_stream_col('SOSEMANUK', 100*1024*1024, 8, 16, 16, 16, eprefix='PH4-SM-25-', streams=g.StreamOptions.CTR_LHW_SAC_RND)

rr += list(itertools.chain.from_iterable(
    [g.generate_block_col('TRIPLE-DES', 100*1024*1024, r, 8, 21, 16, eprefix='PH4-SM-25-', streams=g.StreamOptions.CTR_LHW_SAC_RND) for r in range(1, 17)]))
rr += list(itertools.chain.from_iterable(
    [g.generate_block_col('TRIPLE-DES', 10*1024*1024, r, 8, 21, 16, eprefix='PH4-SM-25-', streams=g.StreamOptions.CTR_LHW_SAC_RND) for r in range(1, 17)]))

to_gen = [
    ('lowmc-s128b', (252, ), [88, 92, 48, 49, 51, 52, 145, 147, 147, 149]),
    ('lowmc-s128c', (None, ), [17, 18]),
]

rr += g.gen_lowmc_core(to_gen, [10 * 1024 * 1024, 100 * 1024 * 1024], eprefix='testmpc25-')

rr += list(itertools.chain.from_iterable(
    [g.generate_hash_inp('Tangle', 100*1024*1024, r, 32, None, None, eprefix='PH4-SM-25-', streams=g.StreamOptions.CTR) for r in range(32, 48)]))

rr += list(itertools.chain.from_iterable(
    [g.generate_hash_inp('Tangle2', 100*1024*1024, r, 32, None, None, eprefix='PH4-SM-25-', streams=g.StreamOptions.CTR) for r in range(33, 48)]))

g.write_submit_obj(rr)
```

# Block cols - test

```python
import os, shutil, itertools
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen30'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]

rr += g.generate_stream_col('RC4', 10*1024*1024, 1, eprefix='PH4-SM-25-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_cfg_inp('stream_cipher', 'RC4', 10*1024*1024, 1, eprefix='PH4-SM-25-', streams=g.StreamOptions.CTR_LHW_SAC_RND)
rr += g.generate_cfg_inp('stream_cipher', 'RC4', 10*1024*1024, 1, eprefix='PH4-SM-25-', streams=g.StreamOptions.CTR_LHW_SAC_RND, key_stream=g.StreamOptions.CTR)
rr += g.generate_cfg_inp('block', 'AES', 10*1024*1024, 1, eprefix='PH4-SM-25-', streams=g.StreamOptions.CTR_LHW_SAC_RND, key_stream=g.StreamOptions.CTR)
rr += g.generate_cfg_inp('hash', 'MD5', 10*1024*1024, 1, eprefix='PH4-SM-25-', streams=g.StreamOptions.CTR_LHW_SAC_RND, key_stream=g.StreamOptions.CTR)
rr += g.generate_cfg_inp('hash', 'SHA3', 10*1024*1024, 1, eprefix='PH4-SM-25-', streams=g.StreamOptions.CTR_LHW_SAC_RND, key_stream=g.StreamOptions.CTR)

g.write_submit_obj(rr)
```


# Tangle top

```python
import os, shutil, itertools
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen33'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]

rr += list(itertools.chain.from_iterable(
    [g.generate_hash_inp('Tangle', 100*1024*1024, r, 32, None, None, eprefix='PH4-SM-25-', streams=g.StreamOptions.CTR) for r in range(64, 81)]))

g.write_submit_obj(rr)
```

# PRNGs

```python
import os, shutil, itertools
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen39'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]

tv_sizes = [32]
osizes = [10*1024*1024, 100*1024*1024, 1000*1024*1024]
prngs = ['std_mersenne_twister', 'std_lcg', 'std_subtract_with_carry', 'testu01-ulcg', 'testu01-umrg', 'testu01-uxorshift']

# zero input, random key, whole stream
kwargs = {'eprefix': 'PH4-SM-39-', 'streams': g.StreamOptions.ZERO, 'key_stream': g.StreamOptions.RND}
for fname, tv_size, osize in itertools.product(prngs, tv_sizes, osizes):
    rr += g.generate_prng_inp(fname, osize, cround=0, tv_size=tv_size, **kwargs)
    
# Re-key with each TV
tv_sizes = [16, 32, 128, 1024, 8192]
kwargs = {'eprefix': 'PH4-SM-39-', 'streams': g.StreamOptions.CTR_LHW_SAC_RND}
for fname, tv_size, osize in itertools.product(prngs, tv_sizes, osizes):
    rr += g.generate_prng_col(fname, osize, cround=0, tv_size=tv_size, **kwargs)

g.write_submit_obj(rr)
```

# RC4 more

```python
import os, shutil, itertools
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen42'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]

rr += g.generate_stream_col('RC4', 1000*1024*1024, 1, eprefix='PH4-SM-42-', streams=g.StreamOptions.SAC_RND)
g.write_submit_obj(rr)
```

# TEA more

```python
import os, shutil, itertools
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen48'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]

rr += list(itertools.chain.from_iterable(
    [g.generate_block_col('TEA', 10*1024*1024, r, eprefix='PH4-SM-48-', streams=g.StreamOptions.LHW) for r in range(27, 33)]))
rr += list(itertools.chain.from_iterable(
    [g.generate_block_col('TEA', 100*1024*1024, r, eprefix='PH4-SM-48-', streams=g.StreamOptions.LHW) for r in range(27, 33)]))
g.write_submit_obj(rr)
```

# LowMC SAC

```python
import os, shutil
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen56'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]

to_gen = [
    ('lowmc-s80a', (None, ), [1, 2, 3, 4]),
    ('lowmc-s80b', (None, ), [1, 2, 3, 4, 5]),
    ('lowmc-s128a', (None, ), [1, 2, 3, 4, 5]),
    ('lowmc-s128b', (None, ), list(range(1, 21)) + [50, 90, 120, 150, 180, 210]),
    ('lowmc-s128c', (None, ), list(range(1, 21)) + [30, 40, 50, 60, 70, 90, 110]),
    ('lowmc-s128d', (None, ), list(range(1, 16)) + [40, 50, 60, 80, 100, 120, 140]),
]

rr += g.gen_lowmc_core(to_gen, [10 * 1024 * 1024, 100 * 1024 * 1024], eprefix='testmpc56-', streams=g.StreamOptions.SAC)
g.write_submit_obj(rr)
```

# LowMC key variant

```python
import os, shutil
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen57'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]

to_gen = [
    ('lowmc-s80a', (None, ), [1, 2, 3, 4]),
    ('lowmc-s80b', (None, ), [1, 2, 3, 4, 5]),
    ('lowmc-s128a', (None, ), [1, 2, 3, 4, 5]),
    ('lowmc-s128b', (None, ), list(range(1, 21)) + [50, 90, 120, 150, 180, 210]),
    ('lowmc-s128c', (None, ), list(range(1, 21)) + [30, 40, 50, 60, 70, 90, 110]),
    ('lowmc-s128d', (None, ), list(range(1, 16)) + [40, 50, 60, 80, 100, 120, 140]),
]

rr += g.gen_lowmc_core(to_gen, [10 * 1024 * 1024, 100 * 1024 * 1024], eprefix='testmpc57-', streams=g.StreamOptions.CTR_LHW_SAC_RND, use_as_key=True)
g.write_submit_obj(rr)
```

# Other MPCs SAC

```python
import os, shutil
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen58'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]

data_sizes = [10 * 1024 * 1024, 100 * 1024 * 1024]
eprefix = 'testmpc58-'
rr += g.gen_posseidon(data_sizes, eprefix, streams=g.StreamOptions.SAC) \
           + g.gen_starkad(data_sizes, eprefix, streams=g.StreamOptions.SAC) \
           + g.gen_rescue(data_sizes, eprefix, streams=g.StreamOptions.SAC) \
           + g.gen_vision(data_sizes, eprefix, streams=g.StreamOptions.SAC) \
           + g.gen_gmimc(data_sizes, eprefix, streams=g.StreamOptions.SAC) \
           + g.gen_mimc(data_sizes, eprefix, streams=g.StreamOptions.SAC)

g.write_submit_obj(rr)
```

# Rescue prime

```python
import os, shutil, itertools
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen63'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]

data_sizes = [10 * 1024 * 1024, 100 * 1024 * 1024]
eprefix = 'testmpc63-'
funcs = ['RescueP_S80a', 'RescueP_S80b', 'RescueP_S80c', 'RescueP_S80d', 'RescueP_128a', 'RescueP_128b', 'RescueP_128c', 'RescueP_128d']
rounds = [1, 2, 3]

rr += list(itertools.chain.from_iterable(
    [g.generate_mpc_cfg(f, data_sizes, r, eprefix=eprefix, streams=g.StreamOptions.CTR_LHW_SAC) for f, r in itertools.product(funcs, rounds)]))

g.write_submit_obj(rr)
```

# Aux

```python
import os, shutil, itertools
from rtt_tools import generator_mpc as g
dname = '/tmp/ggen68'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]

rr += list(itertools.chain.from_iterable(
    [g.generate_stream_col('DECIM', 100*1024*1024, r, eprefix='PH4-SM-68-', streams=g.StreamOptions.LHW) for r in range(4,6)]))
rr += list(itertools.chain.from_iterable(
    [g.generate_block_inp('GOST', 1000*1024*1024, r, eprefix='PH4-SM-68-', streams=g.StreamOptions.LHW) for r in range(29, 33)]))

g.write_submit_obj(rr)
```

# Aux2

```python
import os, shutil, itertools
from rtt_tools import generator_mpc as g
EXP=75
dname = f'/tmp/ggen{EXP}'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]; eprefix=f'PH4-SM-{EXP}-'

rr += g.generate_stream_inp('Grain', 1000*1024*1024, 4, eprefix=eprefix, streams=g.StreamOptions.ZERO)
rr += g.generate_stream_inp('Grain', 1000*1024*1024, 5, eprefix=eprefix, streams=g.StreamOptions.ZERO)
rr += g.generate_stream_col('Grain', 100*1024*1024, 5, eprefix=eprefix, streams=g.StreamOptions.SAC)
rr += g.generate_stream_col('Grain', 100*1024*1024, 6, eprefix=eprefix, streams=g.StreamOptions.SAC)
rr += g.generate_stream_col('Grain', 100*1024*1024, 7, eprefix=eprefix, streams=g.StreamOptions.SAC)
rr += g.generate_stream_col('Grain', 1000*1024*1024, 8, eprefix=eprefix, streams=g.StreamOptions.RND)
rr += g.generate_stream_col('Grain', 1000*1024*1024, 8, eprefix=eprefix, streams=g.StreamOptions.LHW)
rr += g.generate_stream_col('Grain', 1000*1024*1024, 11, eprefix=eprefix, streams=g.StreamOptions.LHW)
rr += g.generate_stream_col('Grain', 1000*1024*1024, 11, eprefix=eprefix, streams=g.StreamOptions.SAC)

# rr += list(itertools.chain.from_iterable(
#     [g.generate_stream_col('DECIM', 100*1024*1024, r, eprefix='PH4-SM-68-', streams=g.StreamOptions.LHW) for r in range(4,6)]))
# rr += list(itertools.chain.from_iterable(
#     [g.generate_block_inp('GOST', 1000*1024*1024, r, eprefix='PH4-SM-68-', streams=g.StreamOptions.LHW) for r in range(29, 33)]))

g.write_submit_obj(rr)
```

# Aux3

```python
import os, shutil, itertools
from rtt_tools import generator_mpc as g
EXP=78
dname = f'/tmp/exps-{EXP}'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]; eprefix=f'PH4-SM-{EXP}-'

rr += g.generate_stream_col('Grain', 100*1024*1024, 5, eprefix=eprefix, streams=g.StreamOptions.RND)
rr += g.generate_stream_col('Grain', 100*1024*1024, 6, eprefix=eprefix, streams=g.StreamOptions.RND)
rr += g.generate_stream_col('Grain', 100*1024*1024, 7, eprefix=eprefix, streams=g.StreamOptions.RND)
rr += g.generate_stream_col('Grain', 1000*1024*1024, 3, eprefix=eprefix, streams=g.StreamOptions.RND)
rr += g.generate_stream_col('Grain', 1000*1024*1024, 8, eprefix=eprefix, streams=g.StreamOptions.SAC)

# rr += list(itertools.chain.from_iterable(
#     [g.generate_stream_col('DECIM', 100*1024*1024, r, eprefix='PH4-SM-68-', streams=g.StreamOptions.LHW) for r in range(4,6)]))
# rr += list(itertools.chain.from_iterable(
#     [g.generate_block_inp('GOST', 1000*1024*1024, r, eprefix='PH4-SM-68-', streams=g.StreamOptions.LHW) for r in range(29, 33)]))

g.write_submit_obj(rr)
```


```python
import os, shutil, itertools
from rtt_tools import generator_mpc as g
EXP=82
dname = f'/tmp/exps-{EXP}'
shutil.rmtree(dname, ignore_errors=True)
os.makedirs(dname, exist_ok=True)
os.chdir(dname); rr=[]; eprefix=f'PH4-SM-{EXP}-'

rr += g.generate_hash_inp('DynamicSHA', 10*1024*1024, 12, eprefix=eprefix, streams=g.StreamOptions.CTR)
g.write_submit_obj(rr)
```


