from io import BufferedReader
import pika
import os
from deepgram import DeepgramClient, PrerecordedOptions
from scipy.io.wavfile import write
import uuid
from openai import OpenAI
from audio import process_audio
from jsonschema import validate
import json
import datetime
import time
import libsql_client

# TODO make into a function so that it can be called asyncronously 
DB_URL = 'file:local.db'

with open('json_schemas/message_input.json', 'r') as file:
    msg_schema = json.loads(file.read())

rabbit_connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
rabbit_channel = rabbit_connection.channel()

voice_exchange = 'voice_buffer'
input_queue_name = 'text_input'

rabbit_channel.exchange_declare(exchange=voice_exchange, exchange_type='topic', durable=True)
voice_queue_declare = rabbit_channel.queue_declare(voice_exchange, durable=True)
input_queue_declare = rabbit_channel.queue_declare(input_queue_name, durable=True)

voice_queue = voice_queue_declare.method.queue
rabbit_channel.queue_bind(
    exchange=voice_exchange,
    queue=voice_queue,
    routing_key='#'
)

def callback(ch, method, properties, body):
    keys = method.routing_key.split('.')
    guild_id = keys[0]
    username = keys[1]
    
    transcriber = ''
    transcriber_key = ''
    with libsql_client.create_client_sync(DB_URL) as client:
        transcriber = str(client.execute("SELECT transcriber FROM familiars WHERE guildId=?", [guild_id]).rows[0][0])
        transcriber_key = str(client.execute("SELECT apikey FROM "+transcriber+" WHERE guildId=?", [guild_id]).rows[0][0])
    
    file_name = str(uuid.uuid4())
    complete_audio = process_audio(body, 48000, 16000, 2)
    write(file_name + '.wav', 16000, complete_audio)
    with open(file_name+".wav", "rb") as audio_file:
        transcription = transcribe(transcriber, transcriber_key, audio_file)
        audio_file.close()
    msg = {
        "guildId": keys[0],
        "text": username + ': ' + transcription,
        "priority": "soft",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
    }
    validate(msg, msg_schema)
    if not transcription.strip() == '': 
        print(transcriber + ' | ' + username + ': ' + transcription)
        rabbit_channel.basic_publish(
            exchange='',
            routing_key=input_queue_name,
            body=json.dumps(msg),
            properties=pika.BasicProperties(
                delivery_mode=pika.DeliveryMode.Persistent
            )
        )
    os.remove(file_name+'.wav')
    ch.basic_ack(delivery_tag=method.delivery_tag)

def transcribe(method: str, key: str, audio_file: BufferedReader, try_count: int = 1) -> str:
    try:
        match method:
            case 'deepgram':
                deepgram = DeepgramClient(key)
                options = PrerecordedOptions(
                    model="nova-2",
                    smart_format=True
                )
                response = deepgram.listen.prerecorded.v("1").transcribe_file({'buffer': audio_file}, options)
                transcription = response['results']['channels'][0]['alternatives'][0]['transcript']
            case 'openai':
                openai = OpenAI(api_key=key)
                response = openai.audio.transcriptions.create(model='whisper-1', file=audio_file)
                transcription = response.text
            case _:
                transcription = ''
    except:
        if try_count <=3 :
            print('Failed transcription, trying again in... ' + str(2**try_count))
            time.sleep(2**try_count)
            transcription = transcribe(method, key, audio_file, try_count+1)
        else:
            print('Failed transcription too many times, returning empty.')
            transcription = ''
    return transcription

# rabbit_channel.basic_qos(prefetch_count=5)
rabbit_channel.basic_consume(
    queue=voice_queue,
    on_message_callback=callback
)
print('Transcription is ready...')
rabbit_channel.start_consuming()
