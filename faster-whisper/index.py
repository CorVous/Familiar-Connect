from faster_whisper import WhisperModel
import sys
import pika
import torch
import io
import numpy
import librosa
import samplerate
import whisper
from scipy.io.wavfile import write
import scipy.signal as sps
import audioop

if torch.cuda.is_available():
    model_size = "medium.en"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    w_model = whisper.load_model('medium.en')
    print("Cuda Available: ", model_size, "loaded...")
else:
    model_size = "base.en"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    w_model = whisper.load_model('base.en')
    print("Cuda Unavailable: ", model_size, "loaded...")

rabbit_connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
rabbit_channel = rabbit_connection.channel()

rabbit_exchange = 'voice_buffer'

rabbit_channel.exchange_declare(exchange='voice_buffer', exchange_type='topic')
result = rabbit_channel.queue_declare('', exclusive=True)
rabbit_queue = result.method.queue
rabbit_channel.queue_bind(
    exchange="voice_buffer",
    queue=rabbit_queue,
    routing_key='#'
)

def callback(ch, method, properties, body):
    print(f" [x] {method.routing_key}")
    audio_buffer = audioop.ratecv(body, 2, 2, 48000, 16000, None)
    audio_buffer = audioop.tomono(audio_buffer[0], 2, 1, 0)
    aaaaa = numpy.frombuffer(audio_buffer, dtype=numpy.int16)
    write('test348.wav', 16000, aaaaa)
    # audio_np = numpy.frombuffer(body, dtype=numpy.int32).astype(numpy.int16)
    # write('test2.wav', 48000, audio_np)
    # audio_buffer = samplerate.resample(audio_np, 16000 * 1.0 / 48000, 'sinc_best')
    # write('test.wav', 16000, audio_buffer)
    # complete_audio = audio_buffer.astype(numpy.float32) / 32768.0
    # num_samples = round(len(audio_np) * float(16000) / 48000)
    # new_audio_np = sps.resample(audio_np, num_samples)
    # write('test3.wav', 16000, new_audio_np)
    complete_audio = aaaaa.astype(numpy.float32) / 32768.0
    result = w_model.transcribe(complete_audio, fp16=torch.cuda.is_available())
    print(f'Regular Whisper: {result['text']}')
    segments, _ = model.transcribe(complete_audio)
    segments = list(segments)
    result = ""
    for segment in segments:
        result += segment.text + ' '
    print("Faster Whisper: " + result)
    ch.basic_ack(delivery_tag=method.delivery_tag)

rabbit_channel.basic_qos(prefetch_count=1)
rabbit_channel.basic_consume(
    queue=rabbit_queue,
    on_message_callback=callback
)
rabbit_channel.start_consuming()
