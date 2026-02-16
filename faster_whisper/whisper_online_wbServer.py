#!/usr/bin/env python3

import os
import logging
import argparse
import asyncio
import websockets
import json
import wave
import io
import numpy as np
from scipy.io import wavfile
import tempfile
import threading
import queue

from .whisper_online import *
from .line_packet import *

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=8765, help="WebSocket server port")
parser.add_argument("--tcp_port", type=int, default=43007, help="Original TCP server port")
parser.add_argument("--warmup_file", type=str, default="tests/data/samples_jfk.wav",
                    help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast.")

# options from whisper_online
add_shared_args(parser)
args = parser.parse_args()

set_logging(args, logger, other="")

# setting whisper object by args
size = args.model
language = args.lan
asr, online = asr_factory(args)
min_chunk = args.min_chunk_size

# warm up the ASR because the very first transcribe takes more time than the others.
msg = "Whisper is not warmed up. The first chunk processing may take longer."
if args.warmup_file:
    if os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file, 0, 1)
        asr.transcribe(a)
        logger.info("Whisper is warmed up.")
    else:
        logger.critical("The warm up file is not available. " + msg)
        import sys

        sys.exit(1)
else:
    logger.warning(msg)


class WebSocketASRProcessor:
    def __init__(self, websocket, asr_model, min_chunk_size):
        self.websocket = websocket
        self.asr_model = asr_model
        self.min_chunk_size = min_chunk_size
        self.audio_buffer = b""
        self.temp_files = []

    async def process_audio_chunk(self, audio_data):
        """处理音频数据块并返回转录结果"""
        try:
            # 添加新音频数据到缓冲区
            self.audio_buffer += audio_data

            # 如果缓冲区数据足够大，则进行转录
            if len(self.audio_buffer) >= self.min_chunk_size:
                # 创建临时WAV文件
                temp_wav = self.create_wav_from_pcm(self.audio_buffer)

                # 使用Whisper模型进行转录
                result = self.asr_model.transcribe(temp_wav)

                # 清空已处理的音频数据
                self.audio_buffer = b""

                # 返回转录结果
                if result and 'text' in result:
                    return result['text']
                else:
                    return ""

        except Exception as e:
            logger.error(f"处理音频块时出错: {e}")
            return f"处理错误: {str(e)}"

    def create_wav_from_pcm(self, pcm_data):
        """将PCM数据转换为WAV格式"""
        try:
            # 假设音频数据是16kHz, 16位, 单声道
            sample_rate = 16000
            dtype = np.int16

            # 将字节数据转换为numpy数组
            audio_array = np.frombuffer(pcm_data, dtype=dtype)

            # 创建临时WAV文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_filename = temp_file.name
            temp_file.close()

            # 保存为WAV文件
            wavfile.write(temp_filename, sample_rate, audio_array)

            # 记录临时文件以便清理
            self.temp_files.append(temp_filename)

            return temp_filename
        except Exception as e:
            logger.error(f"创建WAV文件时出错: {e}")
            raise


async def handle_client(websocket, path):
    """处理WebSocket客户端连接"""
    logger.info(f'新的WebSocket客户端连接: {websocket.remote_address}')

    processor = WebSocketASRProcessor(websocket, asr, min_chunk)

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # 接收到音频数据
                logger.info(f'收到音频数据: {len(message)} 字节')

                # 处理音频数据并获取转录结果
                transcript = await processor.process_audio_chunk(message)

                if transcript:
                    # 发送转录结果
                    response = {
                        "type": "transcript",
                        "text": transcript,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await websocket.send(json.dumps(response, ensure_ascii=False))
                    logger.info(f'发送转录结果: {transcript}')
                else:
                    # 发送确认消息
                    ack_msg = {
                        "type": "ack",
                        "status": "received",
                        "size": len(message)
                    }
                    await websocket.send(json.dumps(ack_msg))
            else:
                # 如果收到文本消息，可能是控制命令
                try:
                    cmd = json.loads(message)
                    if cmd.get("type") == "ping":
                        pong_msg = {"type": "pong"}
                        await websocket.send(json.dumps(pong_msg))
                except json.JSONDecodeError:
                    logger.warning(f'收到未知格式消息: {message}')

    except websockets.exceptions.ConnectionClosed:
        logger.info(f'WebSocket客户端断开连接: {websocket.remote_address}')
    except Exception as e:
        logger.error(f'处理客户端时出错: {e}')
    finally:
        # 清理临时文件
        for temp_file in processor.temp_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass


# 启动WebSocket服务器
logger.info(f'启动WebSocket服务器在 {args.host}:{args.port}')

start_server = websockets.serve(handle_client, args.host, args.port)

logger.info(f'WebSocket服务器监听中: ws://{args.host}:{args.port}')
logger.info(f'原始TCP服务器端口: {args.tcp_port} (保留兼容性)')

# 启动事件循环
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()