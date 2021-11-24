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

