import pika
import numpy
import audioop
import os
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions
from scipy.io.wavfile import write
import uuid
from openai import OpenAI

load_dotenv('../.env')
deepgram = DeepgramClient(os.getenv('DEEPGRAM_KEY') or '')
options = PrerecordedOptions(
    model="nova-2",
    smart_format=True
)
openai = OpenAI(api_key=os.getenv('OPENAI_KEY'))

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
    routing_key='deepgram.#'
)
rabbit_channel.queue_bind(
    exchange="voice_buffer",
    queue=rabbit_queue,
    routing_key='owhisper.#'
)

def callback(ch, method, properties, body):
    key = method.routing_key.split('.')
    print(f" [x] {key[0]}")
    file_name = str(uuid.uuid4())
    audio_buffer = audioop.ratecv(body, 2, 2, 48000, 16000, None)
    audio_buffer = audioop.tomono(audio_buffer[0], 2, 1, 0)
    audio_buffer = audioop.mul(audio_buffer, 2, 5)
    audio_np = numpy.frombuffer(audio_buffer, dtype=numpy.int16)
    complete_audio = audio_np.astype(numpy.float32) / 32768.0
    write(file_name + '.wav', 16000, complete_audio)
    with open(file_name+".wav", "rb") as audio_file:
        match key[0]:
            case 'deepgram':
                response = deepgram.listen.prerecorded.v("1").transcribe_file({'buffer': audio_file}, options)
                transcription = transcription = response['results']['channels'][0]['alternatives'][0]['transcript']
            case 'owhisper':
                response = openai.audio.transcriptions.create(model='whisper-1', file=audio_file)
                transcription = response.text
            case _:
                transcription = ''
        audio_file.close()
    print(transcription)
    
    os.remove(file_name+'.wav')
    ch.basic_ack(delivery_tag=method.delivery_tag)

# rabbit_channel.basic_qos(prefetch_count=5)
rabbit_channel.basic_consume(
    queue=rabbit_queue,
    on_message_callback=callback
)
rabbit_channel.start_consuming()
