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

CHATTINESS = 50
WAIT_TIME = 5
LLM_MODEL = 'claude-3-haiku-20240307'
STARTING_PROMPT = 'You are a cute fox spirit named Sapphire.'

INPUT_QUEUE_NAME = 'text_input'
AUDIO_OUTPUT_EXCHANGE_NAME = 'audio_output'
TEXT_OUTPUT_EXCHANGE_NAME = 'text_output'
load_dotenv()

with open('json_schemas/message_input.json', 'r') as file:
    MSG_SCHEMA = json.loads(file.read())

class MessageProcessor:
    rabbit_connection: pika.BlockingConnection
    rabbit_channel = None
    message_stack: dict
     
    def __init__(self):
        self.rabbit_connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')
        )
        self.message_stack = {}    
        
    async def start(self):
        rabbit_channel = self.rabbit_connection.channel()
        self.rabbit_channel = rabbit_channel
        rabbit_channel.queue_declare(queue=INPUT_QUEUE_NAME, durable=True)
        rabbit_channel.basic_consume(queue=INPUT_QUEUE_NAME, on_message_callback=self.callback)
        rabbit_channel.exchange_declare(exchange=AUDIO_OUTPUT_EXCHANGE_NAME, exchange_type='topic', durable=False)
        rabbit_channel.exchange_declare(exchange=TEXT_OUTPUT_EXCHANGE_NAME, exchange_type='topic', durable=False)
        rabbit_channel.start_consuming()

    def callback(self, ch, method, properties, body):
        msg = json.loads(body.decode())
        validate(msg, MSG_SCHEMA)
        guild_id = msg['guildId']
        if not guild_id in self.message_stack:
            self.message_stack[guild_id] = {}
            self.message_stack[guild_id]['messages'] = []
        elif 'timer' in self.message_stack[guild_id]:
            self.message_stack[guild_id]['timer'].cancel()
        self.message_stack[guild_id]['timer'] = threading.Timer(2, self.process_messages, args=[guild_id])
        self.message_stack[guild_id]['messages'].append(msg)
        
        # TODO: Receive "user is talking" fanout to delay processing
        # will probably need to make this a topic so that it can know what guild ids it's assigned
        # print(f'{msg['guildId']} | {msg['text']}')
        # time.sleep(body.count(b'.'))
        ch.basic_ack(delivery_tag=method.delivery_tag)
        self.message_stack[guild_id]['timer'].start()
    
    def process_messages(self, guild_id: str):
        messages = self.message_stack[guild_id]['messages']
        self.message_stack[guild_id]['messages'] = []
        messages.sort(key=lambda x: datetime.datetime.fromisoformat(x['timestamp']))
         
        response = anthropic_send(STARTING_PROMPT, [], messages, LLM_MODEL, os.getenv('ANTHROPIC_API_KEY'))
        if response == '':
            print('[xx] Empty respnose')
            return
        print('[o] Response ----')
        print(response)
        voice_line = azure_tts(response, 'en-US-AmberNeural', os.getenv('AZURE_KEY'), os.getenv('AZURE_REGION'))
        voice_line, _ = audioop.ratecv(voice_line, 2, 1, 16000, 96000, None)
        # TODO: save history here
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

def anthropic_send(starting_prompt: str, history: list[dict], messages: list[dict], model: str, api_key: str | None) -> str:
    client = anthropic.Anthropic(
        api_key=api_key
    )
    new_message = ''
    for i, msg in enumerate(messages):
        if i < len(messages) - 1:
            new_message += msg['text'] + '\n'
        else:
            new_message += msg['text']
    
    if new_message.strip() == '':
        return ''

    print('[i] New Message ----')
    print(new_message)
    
    new_prompt = []
    
    new_prompt.append({'role': 'user', 'content': new_message})
    
    response = client.messages.create(
        model = model,
        max_tokens=200,
        temperature=0.8,
        system=starting_prompt + "Provide one sentence responses.",
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
