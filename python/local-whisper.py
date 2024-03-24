import pika
import torch
import whisper
from audio import process_audio

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

voice_exchange = 'voice_buffer'
input_queue_name = 'text_input'

rabbit_channel.exchange_declare(exchange='voice_buffer', exchange_type='topic')
voice_queue_declare = rabbit_channel.queue_declare(voice_exchange, durable=True)
input_queue_declare = rabbit_channel.queue_declare(input_queue_name, durable=True)

voice_queue = voice_queue_declare.method.queue
input_queue = input_queue_declare.method.queue
rabbit_channel.queue_bind(
    exchange="voice_buffer",
    queue=voice_queue,
    routing_key='lwhisper.#'
)

def callback(ch, method, properties, body):
    keys = method.routing_key.split('.')
    username = keys[2]
    print(f" [x] {method.routing_key}")
    audio_buffer = process_audio(body, 48000, 16000, 2)
    result = w_model.transcribe(audio_buffer, fp16=torch.cuda.is_available())
    print(f'Regular Whisper: {result['text']}')
    rabbit_channel.basic_publish(
        exchange='',
        routing_key=input_queue_name,
        body=username + ': ' + str(result['text']),
        properties=pika.BasicProperties(
            delivery_mode=pika.DeliveryMode.Persistent
        )
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)

rabbit_channel.basic_qos(prefetch_count=1)
rabbit_channel.basic_consume(
    queue=voice_queue,
    on_message_callback=callback
)
rabbit_channel.start_consuming()
