#!/usr/bin/env python3

import os
import logging
from .whisper_online import *
######### Server objects
from .line_packet import *
import argparse
import time
from fastapi import FastAPI


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# server options
# --language en --model large-v3 --task transcribe
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
SAMPLING_RATE = 16000

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

app = FastAPI()
@app.get("/fileIds/{fileId}")
def read_root(fileId: int):
    filePath = f"tests/data/samples_jfk{fileId}.wav"
    duration = len(load_audio(filePath)) / SAMPLING_RATE
    logger.info("Audio duration is: %2.2f seconds" % duration)
    # load the audio into the LRU cache before we start the timer
    audio = load_audio_chunk(filePath, 0, min_chunk)
    online.insert_audio_chunk(audio)
    text = online.process_iter()
    return "%1.0f %1.0f %s" % (text[0] * 1000, text[1] * 1000, text[2])