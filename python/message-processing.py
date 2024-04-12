from typing import List, Sequence
import pika
import json
import asyncio
import threading
import datetime
import anthropic
from jsonschema import validate
import azure.cognitiveservices.speech as speechsdk
import audioop
import random
import libsql_client

MESSAGE_LIMIT = 100
WAIT_TIME = 2

INPUT_QUEUE_NAME = 'text_input'
AUDIO_OUTPUT_EXCHANGE_NAME = 'audio_output'
TEXT_OUTPUT_EXCHANGE_NAME = 'text_output'
SPEAKING_EXCHANGE_NAME = 'user_speaking'

DB_URL = 'file:local.db'

with open('json_schemas/message_input.json', 'r') as file:
    MSG_SCHEMA = json.loads(file.read())

class MessageProcessor:
    rabbit_connection: pika.BlockingConnection
    rabbit_channel = None
    guild_data: dict[str, dict]

    def __init__(self):
        self.rabbit_connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')
        )
        self.guild_data = {}

    async def start(self):
        rabbit_channel = self.rabbit_connection.channel()
        self.rabbit_channel = rabbit_channel
        rabbit_channel.queue_declare(queue=INPUT_QUEUE_NAME, durable=True)
        rabbit_channel.basic_consume(queue=INPUT_QUEUE_NAME, on_message_callback=self.message_callback)
        rabbit_channel.exchange_declare(exchange=AUDIO_OUTPUT_EXCHANGE_NAME, exchange_type='topic', durable=False)
        rabbit_channel.exchange_declare(exchange=TEXT_OUTPUT_EXCHANGE_NAME, exchange_type='topic', durable=False)
        rabbit_channel.exchange_declare(exchange=SPEAKING_EXCHANGE_NAME, exchange_type='topic', durable=True)
        speaking_queue = rabbit_channel.queue_declare(queue='', exclusive=True).method.queue
        rabbit_channel.queue_bind(exchange=SPEAKING_EXCHANGE_NAME, queue=speaking_queue, routing_key='#')
        rabbit_channel.basic_consume(queue=speaking_queue, on_message_callback=self.speaking_callback)
        rabbit_channel.start_consuming()

    def message_callback(self, ch, method, properties, body):
        msg = json.loads(body.decode())
        validate(msg, MSG_SCHEMA)
        guild_id = msg['guildId']
        if not guild_id in self.guild_data:
            self.guild_data[guild_id] = {}
        if not 'messages' in self.guild_data[guild_id]:
            self.guild_data[guild_id]['messages'] = []

        familiar = []
        with libsql_client.create_client_sync(DB_URL) as client:
            familiar = client.execute("SELECT * FROM familiars WHERE guildId=?", [guild_id]).rows[0]
        familiarName = str(familiar[1])
        staring_prompt = str(familiar[2])
        chattiness = int(str(familiar[3]))
        keywords = str(familiar[4]).split(',')
        llm = str(familiar[6])
        tts = str(familiar[7])
        
        for keyword in keywords:
            if keyword.strip().lower() in msg.get('text').lower():
                print(f'Found {keyword} in message')
                self.guild_data[guild_id].update({'chat_meter': 100})
        if chattiness == 100:
            self.guild_data[guild_id].update({'chat_meter': 100})

        if msg.get('priority') == 'soft':
            if not self.guild_data[guild_id].get('chat_meter'):
                self.guild_data[guild_id]['chat_meter'] = 0
            self.guild_data[guild_id]['chat_meter'] += random.randint(0, int(chattiness / 3))
            # print(guild_id + ' Meter: ' + str(self.guild_data[guild_id]['chat_meter']))
            if (self.guild_data[guild_id]['chat_meter'] >= 75):
                print('meter above 75, now listening')
                print(self.guild_data[guild_id]['chat_meter'])
                self.guild_data[guild_id]['messages'].append(msg)
        else:
            self.guild_data[guild_id]['messages'].append(msg)
        
        if guild_id in self.guild_data and 'timer' in self.guild_data[guild_id]:
            self.guild_data[guild_id]['timer'].cancel()
        if not 'speaking' in self.guild_data[guild_id]:
            self.guild_data[guild_id]['speaking'] = set()
        
        if msg.get('priority') == 'immediate':
            self.guild_data[guild_id]['processing'] = True
            self.process_messages(guild_id)
        elif msg.get('priority') != 'soft' or (msg.get('priority') == 'soft' and (self.guild_data[guild_id].get('chat_meter') or 0) >= 100):
            self.guild_data[guild_id]['processing'] = True
            self.guild_data[guild_id]['timer'] = threading.Timer(WAIT_TIME, self.process_messages, args=[guild_id])
            self.guild_data[guild_id]['timer'].start()

        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    def speaking_callback(self, ch, method, properties, body):
        speaking = body == b'\x01'
        guild_id = method.routing_key.split('.')[0]
        uid = str(method.routing_key.split('.')[1])
        if not guild_id in self.guild_data:
            self.guild_data[guild_id] = {}
        if not 'speaking' in self.guild_data[guild_id]:
            self.guild_data[guild_id]['speaking'] = set()
        if speaking:
            self.guild_data[guild_id]['speaking'].add(uid)
        else:
            if uid in self.guild_data[guild_id]['speaking']:
                self.guild_data[guild_id]['speaking'].remove(uid)
            if len(self.guild_data[guild_id]['speaking']) == 0 and self.guild_data[guild_id].get('processing') == True:
                if 'timer' in self.guild_data[guild_id]:
                    self.guild_data[guild_id]['timer'].cancel()
                self.guild_data[guild_id]['timer'] = threading.Timer(WAIT_TIME, self.process_messages, args=[guild_id])
                self.guild_data[guild_id]['timer'].start()
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def process_messages(self, guild_id: str):
        familiar = []
        llm = []
        tts = []
        with libsql_client.create_client_sync(DB_URL) as client:
            familiar = client.execute("SELECT * FROM familiars WHERE guildId=?", [guild_id]).rows[0]
            llm = client.execute("SELECT * FROM "+ str(familiar[6]) +" WHERE guildId=?", [guild_id]).rows[0]
            tts = client.execute("SELECT * FROM "+ str(familiar[7]) +" WHERE guildId=?", [guild_id]).rows[0]
            history = client.execute("SELECT * FROM history WHERE guildId=? ORDER BY timestamp DESC LIMIT ?", [guild_id, MESSAGE_LIMIT]).rows
        starting_prompt = str(familiar[2])
        llm_type = str(familiar[6])
        tts_type = str(familiar[7])

        self.guild_data[guild_id]['chat_meter'] = 0
        if len(self.guild_data[guild_id]['speaking']) > 0 or self.guild_data[guild_id].get('messages') is None:
            return
        messages = self.guild_data[guild_id]['messages']
        self.guild_data[guild_id]['messages'] = []
        messages.sort(key=lambda x: datetime.datetime.fromisoformat(x['timestamp']))

        new_message = ''
        for i, msg in enumerate(messages):
            if i < len(messages) - 1:
                new_message += msg['text'] + '\n'
            else:
                new_message += msg['text']
        if new_message.strip() == '':
            return
        llm_key = str(llm[1])
        llm_model = str(llm[2])
        llm_tempurature = float(str(llm[3]))
        
        if (llm_type == 'anthropic'):
            response = anthropic_send(starting_prompt, history, new_message, llm_model, llm_key, llm_tempurature)
        else:
            response = 'ummm, not implemented yet, teehee?'

        if response == '':
            print('[xx] Empty respnose')
            return
        print('[o] Response ----')
        print(response)
        voice_response = response.replace("Cor", "Core") # Take this out later. Add a "replace for pronunciation"
        

        tts_key = str(tts[1])
        if tts_type == 'azure':
            azure_region = str(tts[2])
            azure_voice = str(tts[3])
            voice_line = azure_tts(voice_response, azure_voice, tts_key, azure_region)
            voice_line, _ = audioop.ratecv(voice_line, 2, 1, 16000, 96000, None)
        else:
            voice_line = bytes(0)

        with libsql_client.create_client_sync(DB_URL) as client:
            client.execute("INSERT INTO history (guildId, msg, role, timestamp) VALUES (?,?,?,?)", [guild_id, new_message, 'user', datetime.datetime.now().isoformat()])
            client.execute("INSERT INTO history (guildId, msg, role, timestamp) VALUES (?,?,?,?)", [guild_id, response, 'assistant', datetime.datetime.now().isoformat()])

        if self.rabbit_channel:
            self.rabbit_channel.basic_publish(
                exchange=TEXT_OUTPUT_EXCHANGE_NAME,
                routing_key=guild_id,
                body=response
            )
            self.rabbit_channel.basic_publish(
                exchange=AUDIO_OUTPUT_EXCHANGE_NAME,
                routing_key=guild_id,
                body=voice_line
            )
        self.guild_data[guild_id]['processing'] = False

def anthropic_send(starting_prompt: str, history: list, new_message: str, model: str, api_key: str | None, tempurature: float) -> str:
    client = anthropic.Anthropic(
        api_key=api_key
    )

    print('[i] New Message ----')
    print(new_message)

    new_prompt = []
    
    for entry in list(reversed(history)):
        guild_id, msg, img, role, timestamp = str(entry[0]), str(entry[1]), str(entry[2]), str(entry[3]), str(entry[4])
        new_prompt.append({'role': role, 'content': msg})

    new_prompt.append({'role': 'user', 'content': new_message})

    response = client.messages.create(
        model = model,
        max_tokens=200,
        temperature=tempurature,
        system=starting_prompt,
        messages=new_prompt
    )
    client.close()
    return response.content[0].text

def azure_tts(text: str, voice_name: str, api_key: str | None, region: str | None) -> bytes:
    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    speech_config.speech_synthesis_voice_name = voice_name
    audio_config = speechsdk.audio.PullAudioOutputStream()
    speech_synthesizer = speechsdk.SpeechSynthesizer(audio_config=audio_config,speech_config=speech_config) # type: ignore
    result: speechsdk.SpeechSynthesisResult = speech_synthesizer.speak_text(text)
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        stream = result.audio_data
        return stream[44:]
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print(f"Error details: {cancellation_details.error_details}")
                print(f"Did you set the speech resource key and region values?")
    return bytes(0)

if (__name__ == "__main__"):
    owo = MessageProcessor()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = asyncio.gather(owo.start())
    try:
        print('starting...')
        loop.run_until_complete(tasks)
    except KeyboardInterrupt as e:
        print('closing...')
        tasks.cancel()
    finally:
        loop.close()
