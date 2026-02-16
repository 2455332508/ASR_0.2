#!/usr/bin/env python3
import logging
import sys
import time
from functools import lru_cache

import librosa
import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(10 ** 6)
def load_audio(fname):
    audioDataArr, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return audioDataArr

def load_audio_chunk(fname, beg, end):
    audioDataArr = load_audio(fname)
    beg_s = int(beg * 16000)
    end_s = int(end * 16000)
    return audioDataArr[beg_s:end_s]

def set_logging(args, logger, other="_server"):
    logging.basicConfig(  # format='%(name)s
        format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)
    logging.getLogger("whisper_online" + other).setLevel(args.log_level)
#    logging.getLogger("whisper_online_server").setLevel(args.log_level)

def add_shared_args(parser):
    """shared args for simulation (this entry point) and server
    parser: argparse.ArgumentParser object
    """
    parser.add_argument('--min-chunk-size', type=float, default=1.0,
                        help='Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.')
    parser.add_argument('--model', type=str, default='large-v3',
                        choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo".split(
                            ","),
                        help="Name size of the Whisper model to use (default: large-v2). The model is automatically downloaded from the model hub if not present in model cache dir.")
    parser.add_argument('--model_cache_dir', type=str, default=None,
                        help="Overriding the default model cache dir where models downloaded from the hub are saved")
    parser.add_argument('--model_dir', type=str, default=None,
                        help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.")
    parser.add_argument('--lan', '--language', type=str, default='en',
                        help="Source language code, e.g. en,de,cs, or 'auto' for language detection.")
    parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe", "translate"],
                        help="Transcribe or translate.")
    parser.add_argument('--backend', type=str, default="faster-whisper",
                        choices=["faster-whisper", "whisper_timestamped", "mlx-whisper", "openai-api"],
                        help='Load only this backend for Whisper processing.')
    parser.add_argument('--vac', action="store_true", default=False, # todo true
                        help='Use VAC = voice activity controller. Recommended. Requires torch.')
    parser.add_argument('--vac-chunk-size', type=float, default=0.04, help='VAC sample size in seconds.')
    parser.add_argument('--vad', action="store_true", default=False,
                        help='Use VAD = voice activity detection, with the default parameters.')
    parser.add_argument('--buffer_trimming', type=str, default="segment", choices=["sentence", "segment"],
                        help='Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.')
    parser.add_argument('--buffer_trimming_sec', type=float, default=15,
                        help='Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.')
    parser.add_argument("-l", "--log-level", dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the log level",
                        default='DEBUG')

def asr_factory(args, logfile=sys.stderr):
    """
    Creates and configures an ASR and ASR Online instance based on the specified backend and arguments.
    """
    backend = args.backend
    if backend == "openai-api":
        logger.debug("Using OpenAI API.")
        # asr = OpenaiApiASR(lan=args.lan)
    else:
        from .WhisperBackend.FasterWhisperASR import FasterWhisperASR
        # Only for FasterWhisperASR and WhisperTimestampedASR
        size = args.model
        t = time.time()
        logger.info(f"Loading Whisper {size} model for {args.lan}...")
        asr = FasterWhisperASR(lan=args.lan, modelsize=size, cache_dir=args.model_cache_dir, model_dir=args.model_dir)
        e = time.time()
        logger.info(f"done. It took {round(e - t, 2)} seconds.")

    # Apply common configurations
    if getattr(args, 'vad', False):  # Checks if VAD argument is present and True
        logger.info("Setting VAD filter")
        asr.use_vad()

    language = args.lan
    if args.task == "translate":
        asr.set_translate_task()
        tgt_language = "zh"  # Whisper translates into Chinese
    else:
        tgt_language = language  # Whisper transcribes in this language

    # Create the tokenizer , removed by zt since it can't be installed
    tokenizer = None
    # Create the ASRProcessor
    if args.vac:
        from .ASRProcessor import VACOnlineASRProcessor
        online = VACOnlineASRProcessor(args.min_chunk_size, asr, tokenizer, logfile=logfile,
                                       buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))
    else:
        from .ASRProcessor import OnlineASRProcessor
        online = OnlineASRProcessor(asr, tokenizer, logfile=logfile,
                                    buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))
    return asr, online

def output_transcript(o, now=None):
    # output format in stdout is like:
    # 4186.3606 0 1720 Takhle to je
    # - the first three words are:
    #    - emission time from beginning of processing, in milliseconds
    #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
    # - the next words: segment transcript
    if now is None:
        now = time.time() - start
        return None
    if o[0] is not None:
        print("%1.4f %1.0f %1.0f %s" % (now * 1000, o[0] * 1000, o[1] * 1000, o[2]), file=logfile, flush=True)
        print("%1.4f %1.0f %1.0f %s" % (now * 1000, o[0] * 1000, o[1] * 1000, o[2]), flush=True)
        return "%1.4f %1.0f %1.0f %s" % (now * 1000, o[0] * 1000, o[1] * 1000, o[2])
    else:
        # No text, so no output
        pass

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str,
                        help="Filename of 16kHz mono channel wav, on which live streaming is simulated.")
    add_shared_args(parser)
    parser.add_argument('--start_at', type=float, default=0.0, help='Start processing audio at this time.')
    parser.add_argument('--offline', action="store_true", default=False, help='Offline mode.')
    parser.add_argument('--comp_unaware', action="store_true", default=False,
                        help='Computationally unaware simulation.')

    args = parser.parse_args()

    # reset to store stderr to different file stream, e.g. open(os.devnull,"w")
    logfile = sys.stderr

    if args.offline and args.comp_unaware:
        logger.error("No or one option from --offline and --comp_unaware are available, not both. Exiting.")
        sys.exit(1)

    #    if args.log_level:
    #        logging.basicConfig(format='whisper-%(levelname)s:%(name)s: %(message)s',
    #                            level=getattr(logging, args.log_level))

    set_logging(args, logger)

    audio_path = args.audio_path

    SAMPLING_RATE = 16000
    duration = len(load_audio(audio_path)) / SAMPLING_RATE
    logger.info("Audio duration is: %2.2f seconds" % duration)

    asr, online = asr_factory(args, logfile=logfile)
    if args.vac:
        min_chunk = args.vac_chunk_size
    else:
        min_chunk = args.min_chunk_size

    # load the audio into the LRU cache before we start the timer
    a = load_audio_chunk(audio_path, 0, 1)

    # warm up the ASR because the very first transcribe takes much more time than the other
    asr.transcribe(a)

    beg = args.start_at
    start = time.time() - beg

    if args.offline:  ## offline mode processing (for testing/debugging)
        a = load_audio(audio_path)
        online.insert_audio_chunk(a)
        try:
            o = online.process_iter()
        except AssertionError as e:
            logger.error(f"assertion error: {repr(e)}")
        else:
            output_transcript(o)
        now = None
    elif args.comp_unaware:  # computational unaware mode 
        end = beg + min_chunk
        while True:
            a = load_audio_chunk(audio_path, beg, end)
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {repr(e)}")
                pass
            else:
                output_transcript(o, now=end)

            logger.debug(f"## last processed {end:.2f}s")

            if end >= duration:
                break

            beg = end

            if end + min_chunk > duration:
                end = duration
            else:
                end += min_chunk
        now = duration
    else:  # online = simultaneous mode
        end = 0
        while True:
            now = time.time() - start
            if now < end + min_chunk:
                time.sleep(min_chunk + end - now)
            end = time.time() - start
            a = load_audio_chunk(audio_path, beg, end)
            beg = end
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {e}")
                pass
            else:
                output_transcript(o)
            now = time.time() - start
            logger.debug(f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now - end:.2f}")

            if end >= duration:
                break
        now = None
    o = online.finish()
    output_transcript(o, now=now)
