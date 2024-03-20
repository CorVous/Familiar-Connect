from faster_whisper import WhisperModel
import sys
import pika
import torch
import io
import numpy

if False:
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    print("Cuda Available: ", model_size, "loaded...")
else:
    model_size = "base.en"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
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
    audio_buffer = numpy.frombuffer(body, dtype=numpy.int16).astype(numpy.float32)
    segments, _ = model.transcribe(audio_buffer)
    segments = list(segments)  # The transcription will actually run here.
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

rabbit_channel.basic_consume(
    queue=rabbit_queue,
    on_message_callback=callback,
    auto_ack=True
)
rabbit_channel.start_consuming()