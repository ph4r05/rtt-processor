#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import os
import binascii
from rtt_tools.dump_data import Loader


class RttDumpTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(RttDumpTest, self).__init__(*args, **kwargs)

    def test_ph4_sm(self):
        l = Loader()
        r = l.break_exp_ph4('PH4-SM-02-testu01-uxorshift-t:prng-r:x-b:400-s:10MiB-e:2-i:hw-256bit-hw3-offsetidx-1842346-offset-78-96-243-r0.67-vecsize-921174-10MiB')
        self.assertEqual(r.fnc, 'testu01-uxorshift')
        self.assertEqual(r.osize, 10485760)

        r = l.break_exp_ph4('PH4-SM-01-Tangle2-t:hash-r:25-b:32-s:100MiB-e:2-i:hw-256bit-hw4-offsetidx-116528426-offset-61-70-84-173-r0.67-vecsize-58264214-100MiB')
        self.assertEqual(r.fnc, 'Tangle2')
        self.assertEqual(r.osize, 104857600)

        r = l.break_exp_ph4('PH4-SM-04-F-FCSR-t:stream_cipher-r:1-b:16-s:100MiB-e:0-i:ctr-128sbit-offset-0')
        self.assertEqual(r.fnc, 'F-FCSR')
        self.assertEqual(r.osize, 104857600)

        r = l.break_exp_ph4('PH4-SM-07-AES-t:block-r:5-b:16-s:100MiB-e:0-i:ctr.key-128sbit-offset-0')
        self.assertEqual(r.fnc, 'AES')
        self.assertEqual(r.osize, 104857600)

        r = l.break_exp_ph4('PH4-SM-07-F-FCSR-t:stream_cipher-r:1-b:16-s:100MiB-e:0-i:ctr.key-128sbit-offset-0')
        self.assertEqual(r.fnc, 'F-FCSR')
        self.assertEqual(r.osize, 104857600)

        r = l.break_exp_ph4('PH4-SM-29-AES-t:block-r:2-b:16-s:20MiB-e:2-i:ctr.key..inp.ctr-128sbit-offset-2')
        self.assertEqual(r.fnc, 'AES')
        self.assertEqual(r.osize, 20 * 1024 * 1024)

        r = l.break_exp_ph4('PH4-SM-14-AES-t:block-r:1-b:16-s:1000MiB-e:1-i:ctr.key-128sbit-offset-1')
        self.assertEqual(r.fnc, 'AES')
        self.assertEqual(r.osize, 1000 * 1024 * 1024)

        r = l.break_exp_ph4('PH4-SM-59-XTEA-t:block-r:2-b:8-s:10MiB-e:2-i:sac.key..ornd.inp-128sbit-offset-2')
        self.assertEqual(r.fnc, 'XTEA')
        self.assertEqual(r.osize, 10 * 1024 * 1024)

        r = l.break_exp_ph4('PH4-SM-82-DynamicSHA-t:hash-r:12-b:64-s:10MiB-e:1-i:hw-hw-512bit-hw3-offsetidx-7412906-offset-64-229-334-r0.33-vecsize-14825814-10.00MB')
        self.assertEqual(r.fnc, 'DynamicSHA')
        self.assertEqual(r.osize, 10 * 1024 * 1024)

    def test_mpc(self):
        l = Loader()
        r1 = l.break_exp_ph4_mpc('testmpc02-lowmc-s128b-bin-raw-r6-inp-ctr00-b16-spr--s100MB.json')
        self.assertEqual(r1.fnc, 'lowmc-s128b')
        self.assertEqual(r1.osize, 104857600)

        r2 = l.break_exp_ph4_mpc('testmpc01-Poseidon_S80b-pri-raw-r1-0-0-inp-ctr02-b20-spr-s15ob-s10MB.json')
        self.assertEqual(r2.fnc, 'Poseidon_S80b')
        self.assertEqual(r2.osize, 10485760)
        self.assertEqual(r2.spread, 's15ob')

        r3 = l.break_exp_ph4_mpc('testmpc01-Vision_S128d-bin-raw-r1-inp-lhw02-b7-w7-spr-s18ob-s100MB.json')
        self.assertEqual(r3.fnc, 'Vision_S128d')
        self.assertEqual(r3.osize, 104857600)

        r4 = l.break_exp_ph4_mpc('testmpc01-S45-pri-raw-r1-inp-ctr01-b11-spr-s6mb-s100MB.json')
        self.assertEqual(r4.fnc, 'S45')
        self.assertEqual(r4.osize, 104857600)

        r5 = l.break_exp_ph4_mpc('starkad_poseidon-testmpc01-Poseidon_S80b-pri-raw-r1-0-0-inp-lhw01-b20-w5-spr-s15mb-s100MB.json')
        self.assertEqual(r5.fnc, 'Poseidon_S80b')
        self.assertEqual(r5.osize, 104857600)

        r6 = l.break_exp_ph4_mpc('mimc_hash-testmpc01-S45-pri-raw-r2-inp-ctr01-b11-spr-s6ob-s100MB.json')
        self.assertEqual(r6.fnc, 'mimc_hash-S45')
        self.assertEqual(r6.osize, 104857600)

        r = l.break_exp_ph4_mpc('testmpc63-RescueP_128d-pri-raw-r3-inp-lhw01-b31-w4-spr-s16mb1-s100MB')
        self.assertEqual(r.osize, 104857600)

        r = l.break_exp_ph4_mpc('testmpc58-S80-pri-raw-r1-inp-sac00-b20-spr-s15mb-s100MB-boolex-162-324-486')
        self.assertEqual(r.osize, 104857600)

        r = l.break_exp_ph4_mpc('testmpc57-lowmc-s128d-bin-raw-r6-inp-ctr01.key-b16-spr--s10MB')
        self.assertEqual(r.osize, 10485760)
        self.assertEqual(r.meth, 'ctr.key')

        r = l.break_exp_ph4_mpc('testmpc57-lowmc-s128d-bin-raw-r6-inp-ctr01.key..hw.inp-b16-spr--s10MB')
        self.assertEqual(r.osize, 10485760)
        self.assertEqual(r.meth, 'ctr.key..hw.inp')


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
