#!/usr/bin/env python3

import os
import logging
from .whisper_online import *
######### Server objects
from .line_packet import *
import socket


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=43007)
parser.add_argument("--warmup_file", type=str, default="tests/data/samples_jfk.wav",
        help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")

# options from whisper_online
add_shared_args(parser)
args = parser.parse_args()

set_logging(args,logger,other="")

# setting whisper object by args
size = args.model
language = args.lan
asr, online = asr_factory(args)
min_chunk = args.min_chunk_size

# warm up the ASR because the very first transcribe takes more time than the others.
# Test results in https://github.com/ufal/whisper_streaming/pull/81
msg = "Whisper is not warmed up. The first chunk processing may take longer."
if args.warmup_file:
    if os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file,0,1)
        asr.transcribe(a)
        logger.info("Whisper is warmed up.")
    else:
        logger.critical("The warm up file is not available. "+msg)
        sys.exit(1)
else:
    logger.warning(msg)
# server loop

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((args.host, args.port))
    s.listen(1)
    logger.info('Listening on'+str((args.host, args.port)))
    while True:
        conn, addr = s.accept()
        logger.info('Connected to client on {}'.format(addr))
        connection = Connection(conn)
        proc = ServerProcessor(connection, online, args.min_chunk_size)
        proc.process()
        conn.close()
        logger.info('Connection to client closed')
logger.info('Connection closed, terminating.')

