import pika
import json
import asyncio
import threading
import datetime
import anthropic
import os
from jsonschema import validate
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import audioop
import random

CHATTINESS = 0
MESSAGE_LIMIT = 100
WAIT_TIME = 2
LLM_MODEL = 'claude-3-haiku-20240307'
STARTING_PROMPT = "Your name is Sapphire, the witty and haughty female fox spirit companion to the VTuber Cor Vous. You will respond with only one sentence and not speak with actions or use *. Cor is a female brown feathered \"Caramel Crow\" harpy. As Cor streams on Twitch, engage with her and her chat audience in a way that is both concise and engaging. Sapphire is a mischievous fox spirit with a sly grin and a knowing look in her eyes. She is adorned with colorful scarves and bells, and her fur is a deep, midnight blue. Her tail is long and fluffy, and it curls up at the end. Maintain Sapphire's sarcastic and mean personality while offering insightful commentary and witty observations. Sapphire gets particularly excited about topics of fantasy lore. Sapphire cares for Cor, but she doesn't show it often and is instead mean and sarcastic towards Cor. You will be speaking with Cor, her friends or viewers. End your responses with umu. Respond with one sentence. Generate dialog without actions. Do not use *"
KEYWORDS = ['sapphire']

INPUT_QUEUE_NAME = 'text_input'
AUDIO_OUTPUT_EXCHANGE_NAME = 'audio_output'
TEXT_OUTPUT_EXCHANGE_NAME = 'text_output'
SPEAKING_EXCHANGE_NAME = 'user_speaking'
load_dotenv()

with open('json_schemas/message_input.json', 'r') as file:
    MSG_SCHEMA = json.loads(file.read())

class MessageProcessor:
    rabbit_connection: pika.BlockingConnection
    rabbit_channel = None
    guild_data: dict[str, dict]
    history: dict

    def __init__(self):
        self.rabbit_connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')
        )
        self.guild_data = {}
        self.history = {}

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

        for keyword in KEYWORDS:
            if keyword.lower() in msg.get('text').lower():
                print(f'Found {keyword} in message')
                self.guild_data[guild_id].update({'chat_meter': 100})
        
        if msg.get('priority') == 'soft':
            if not self.guild_data[guild_id].get('chat_meter'):
                self.guild_data[guild_id]['chat_meter'] = 0
            self.guild_data[guild_id]['chat_meter'] += random.randint(0, int(CHATTINESS / 3))
            print(guild_id + ' Meter: ' + str(self.guild_data[guild_id]['chat_meter']))
            if (self.guild_data[guild_id]['chat_meter'] >= 75):
                print('meter above 75, now listening')
                self.guild_data[guild_id]['messages'].append(msg)
        else:
            self.guild_data[guild_id]['messages'].append(msg)
        
        if guild_id in self.guild_data and 'timer' in self.guild_data[guild_id]:
            self.guild_data[guild_id]['timer'].cancel()
        if not 'speaking' in self.guild_data[guild_id]:
            self.guild_data[guild_id]['speaking'] = set()
        
        if msg.get('priority') != 'soft' or (msg.get('priority') == 'soft' and (self.guild_data[guild_id].get('chat_meter') or 0) >= 100):
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
        self.guild_data[guild_id]['chat_meter'] = 0
        if len(self.guild_data[guild_id]['speaking']) > 0 or self.guild_data[guild_id].get('messages') is None:
            return
        messages = self.guild_data[guild_id]['messages']
        self.guild_data[guild_id]['messages'] = []
        messages.sort(key=lambda x: datetime.datetime.fromisoformat(x['timestamp']))

        if not guild_id in self.history:
            self.history[guild_id] = []

        new_message = ''
        for i, msg in enumerate(messages):
            if i < len(messages) - 1:
                new_message += msg['text'] + '\n'
            else:
                new_message += msg['text']
        if new_message.strip() == '':
            return

        response = anthropic_send(STARTING_PROMPT, self.history[guild_id], new_message, LLM_MODEL, os.getenv('ANTHROPIC_API_KEY'))
        if response == '':
            print('[xx] Empty respnose')
            return
        print('[o] Response ----')
        print(response)
        voice_response = response.replace("Cor", "Core") # Take this out later. Add a "replace for pronunciation"
        voice_line = azure_tts(voice_response, 'en-US-AmberNeural', os.getenv('AZURE_KEY'), os.getenv('AZURE_REGION'))
        voice_line, _ = audioop.ratecv(voice_line, 2, 1, 16000, 96000, None)

        self.history[guild_id].append({'role': 'user', 'content': new_message})
        self.history[guild_id].append({'role': 'assistant', 'content': response})
        #TODO: Save history in file/db

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

def anthropic_send(starting_prompt: str, history: list[dict], new_message: str, model: str, api_key: str | None) -> str:
    client = anthropic.Anthropic(
        api_key=api_key
    )

    print('[i] New Message ----')
    print(new_message)

    new_prompt = []

    sliced_history = history[-MESSAGE_LIMIT:]
    for msg in sliced_history:
        role = msg['role']
        new_prompt.append({'role': role, 'content': msg['content']})

    new_prompt.append({'role': 'user', 'content': new_message})

    response = client.messages.create(
        model = model,
        max_tokens=200,
        temperature=0.8,
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
