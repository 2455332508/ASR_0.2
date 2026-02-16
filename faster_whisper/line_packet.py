#!/usr/bin/env python3

"""Functions for sending and receiving individual lines of text over a socket.

A line is transmitted using one or more fixed-size packets of UTF-8 bytes
containing:

  - Zero or more bytes of UTF-8, excluding \n and \0, followed by

  - Zero or more \0 bytes as required to pad the packet to PACKET_SIZE

Originally from the UEDIN team of the ELITR project. 
"""
import io
import soundfile
import sys
import numpy as np
import logging
import librosa

PACKET_SIZE = 65536
SAMPLING_RATE = 16000
logger = logging.getLogger(__name__)



def send_one_line(socket, text, pad_zeros=False):
    """Sends a line of text over the given socket.

    The 'text' argument should contain a single line of text (line break
    characters are optional). Line boundaries are determined by Python's
    str.splitlines() function [1]. We also count '\0' as a line terminator.
    If 'text' contains multiple lines then only the first will be sent.

    If the send fails then an exception will be raised.

    [1] https://docs.python.org/3.5/library/stdtypes.html#str.splitlines

    Args:
        socket: a socket object.
        text: string containing a line of text for transmission.
    """
    text.replace('\0', '\n')
    lines = text.splitlines()
    first_line = '' if len(lines) == 0 else lines[0]
    # TODO Is there a better way of handling bad input than 'replace'?
    data = (first_line.encode('utf-8', errors='replace')
            + b'\n' + (b'\0' if pad_zeros else b''))
    for offset in range(0, len(data), PACKET_SIZE):
        bytes_remaining = len(data) - offset
        if bytes_remaining < PACKET_SIZE:
            padding_length = PACKET_SIZE - bytes_remaining
            packet = data[offset:] + (b'\0' * padding_length if pad_zeros else b'')
        else:
            packet = data[offset:offset+PACKET_SIZE]
        socket.sendall(packet)


def receive_one_line(socket):
    """Receives a line of text from the given socket.

    This function will (attempt to) receive a single line of text. If data is
    currently unavailable then it will block until data becomes available or
    the sender has closed the connection (in which case it will return an
    empty string).

    The string should not contain any newline characters, but if it does then
    only the first line will be returned.

    Args:
        socket: a socket object.

    Returns:
        A string representing a single line with a terminating newline or
        None if the connection has been closed.
    """
    data = b''
    while True:
        packet = socket.recv(PACKET_SIZE)
        if not packet:  # Connection has been closed.
            return None
        data += packet
        if b'\0' in packet:
            break
    # TODO Is there a better way of handling bad input than 'replace'?
    text = data.decode('utf-8', errors='replace').strip('\0')
    lines = text.split('\n')
    return lines[0] + '\n'


def receive_lines(socket):
    try:
        data = socket.recv(PACKET_SIZE)
    except BlockingIOError:
        return []
    if data is None:  # Connection has been closed.
        return None
    # TODO Is there a better way of handling bad input than 'replace'?
    text = data.decode('utf-8', errors='replace').strip('\0')
    lines = text.split('\n')
    if len(lines)==1 and not lines[0]:
        return None
    return lines



class Connection:
    '''it wraps conn object'''
    PACKET_SIZE = 32000*5*60 # 5 minutes # was: 65536

    def __init__(self, conn):
        self.conn = conn
        self.last_line = ""

        self.conn.setblocking(True)

    def send(self, line):
        '''it doesn't send the same line twice, because it was problematic in online-text-flow-events'''
        if line == self.last_line:
            return
        send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        in_line = receive_lines(self.conn)
        return in_line

    def non_blocking_receive_audio(self):
        try:
            r = self.conn.recv(self.PACKET_SIZE)
            return r
        except ConnectionResetError:
            return None


# wraps socket and ASR object, and serves one client connection.
# next client should be served by a new instance of this object
class ServerProcessor:

    def __init__(self, c, online_asr_proc, min_chunk):
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk

        self.last_end = None

        self.is_first = True

    def receive_audio_chunk(self):
        # receive all audio that is available by this time
        # blocks operation if less than self.min_chunk seconds is available
        # unblocks if connection is closed or a chunk is available
        out = []
        minlimit = self.min_chunk*SAMPLING_RATE
        while sum(len(x) for x in out) < minlimit:
            raw_bytes = self.connection.non_blocking_receive_audio()
            if not raw_bytes:
                break
#            print("received audio:",len(raw_bytes), "bytes", raw_bytes[:10])
            sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1,
                                     endian="LITTLE",samplerate=SAMPLING_RATE,
                                     subtype="PCM_16",format="RAW")
            audio, _ = librosa.load(sf,sr=SAMPLING_RATE,dtype=np.float32)
            out.append(audio)
        if not out:
            return None
        conc = np.concatenate(out)
        if self.is_first and len(conc) < minlimit:
            return None
        self.is_first = False
        return np.concatenate(out)

    def format_output_transcript(self,o):
        # output format in stdout is like:
        # 0 1720 Takhle to je
        # - the first two words are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript

        # This function differs from whisper_online.output_transcript in the following:
        # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
        # Therefore, beg, is max of previous end and current beg outputed by Whisper.
        # Usually it differs negligibly, by appx 20 ms.

        if o[0] is not None:
            beg, end = o[0]*1000,o[1]*1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            print("%1.0f %1.0f %s" % (beg,end,o[2]),flush=True,file=sys.stderr)
            return "%1.0f %1.0f %s" % (beg,end,o[2])
        else:
            logger.debug("No text in this segment")
            return None

    def send_result(self, o):
        msg = self.format_output_transcript(o)
        if msg is not None:
            self.connection.send(msg)

    def process(self):
        # handle one client connection
        self.online_asr_proc.init()
        while True:
            a = self.receive_audio_chunk()
            if a is None:
                break
            self.online_asr_proc.insert_audio_chunk(a)
            o = self.online_asr_proc.process_iter()
            try:
                self.send_result(o)
            except BrokenPipeError:
                logger.info("broken pipe -- connection closed?")
                break

#        o = online.finish()  # this should be working
#        self.send_result(o)