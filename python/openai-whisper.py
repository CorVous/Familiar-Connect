import pika
import numpy
import audioop
import os
from dotenv import load_dotenv
from scipy.io.wavfile import write
import uuid
from openai import OpenAI

load_dotenv('../.env')

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
    file_name = str(uuid.uuid4())
    audio_np = numpy.frombuffer(audio_buffer, dtype=numpy.int16)
    complete_audio = audio_np.astype(numpy.float32) / 32768.0
    write(file_name + '.wav', 16000, complete_audio)
    translation = 
    print(response.to_json(indent=4))
    # print(f'Regular Whisper: {result['text']}')
    ch.basic_ack(delivery_tag=method.delivery_tag)

rabbit_channel.basic_qos(prefetch_count=1)
rabbit_channel.basic_consume(
    queue=rabbit_queue,
    on_message_callback=callback
)
rabbit_channel.start_consuming()
