import pika
import torch
import numpy
import whisper
import audioop

if torch.cuda.is_available():
    model_size = 'medium.en'
    w_model = whisper.load_model(model_size)
    print("Cuda Available: ", model_size, "loaded...")
else:
    model_size = 'base.en'
    w_model = whisper.load_model(model_size)
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
    routing_key='lwhisper.#'
)

def callback(ch, method, properties, body):
    print(f" [x] {method.routing_key}")
    audio_buffer = audioop.ratecv(body, 2, 2, 48000, 16000, None)
    audio_buffer = audioop.tomono(audio_buffer[0], 2, 1, 0)
    audio_buffer = audioop.mul(audio_buffer, 2, 5)
    audio_buffer = numpy.frombuffer(audio_buffer, dtype=numpy.int16)
    audio_buffer = audio_buffer.astype(numpy.float32) / 32768.0
    result = w_model.transcribe(audio_buffer, fp16=torch.cuda.is_available())
    print(f'Regular Whisper: {result['text']}')
    ch.basic_ack(delivery_tag=method.delivery_tag)

rabbit_channel.basic_qos(prefetch_count=1)
rabbit_channel.basic_consume(
    queue=rabbit_queue,
    on_message_callback=callback
)
rabbit_channel.start_consuming()
