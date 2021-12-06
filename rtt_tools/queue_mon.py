#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import json
import math
import logging
import coloredlogs
import argparse
from rtt_tools.common.rtt_db_conn import *
import configparser

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class QueueMon:
    def __init__(self):
        self.args = None
        self.conn = None

    def arg_parse(self):
        parser = argparse.ArgumentParser(description='QueueMon')
        parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                            help='enables debug mode')
        parser.add_argument('--print', dest='print', action='store_const', const=True,
                            help='Print status')
        parser.add_argument('--out', dest='out', default='queue_mon.json',
                            help='Output record file')
        self.args, unparsed = parser.parse_known_args()
        logger.debug("Unparsed: %s" % unparsed)

    def connect(self):
        cfg = configparser.ConfigParser()
        cfg.read("config.ini")
        self.conn = create_mysql_db_conn(cfg)

    def work(self):
        self.arg_parse()
        self.connect()
        if self.args.debug:
            coloredlogs.install(level=logging.DEBUG, use_chroot=False)

        sql_sel = "SELECT count(*), status FROM jobs WHERE status in ('running', 'pending') group by status order by status"
        with self.conn.cursor() as c:
            c.execute(sql_sel)
            dt = collections.OrderedDict([
                ('pending', 0),
                ('running', 0),
            ])

            for result in c.fetchall():
                cnt, name = result[0], result[1]
                dt[name] = int(cnt)

            js = collections.OrderedDict([
                ('time', int(time.time()*1000)),
                ('running', dt['running']),
                ('pending', dt['pending']),
            ])

            with open(self.args.out, 'a+') as fh:
                json.dump(js, fh)
                fh.write('\n')

            if self.args.print:
                print(f'{json.dumps(js, indent=2)}')


def main():
    QueueMon().work()


if __name__ == '__main__':
    main()
