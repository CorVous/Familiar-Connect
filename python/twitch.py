import time
from twitchAPI.helper import first
from twitchAPI.twitch import Twitch, TwitchUser
from twitchAPI.oauth import UserAuthenticationStorageHelper
from twitchAPI.object.eventsub import ChannelPointsCustomRewardRedemptionAddEvent, ChannelSubscribeEvent, ChannelSubscriptionGiftEvent, ChannelSubscriptionMessageEvent, ChannelCheerEvent, ChannelFollowEvent, ChannelAdBreakBeginEvent
from twitchAPI.eventsub.websocket import EventSubWebsocket
from twitchAPI.type import AuthScope
import asyncio
import threading
import pika
import json
from jsonschema import validate
import os
from dotenv import load_dotenv
import datetime

load_dotenv('.env')
TARGET_SCOPES = [AuthScope.CHANNEL_READ_REDEMPTIONS, AuthScope.CHANNEL_READ_SUBSCRIPTIONS,  AuthScope.BITS_READ, AuthScope.MODERATOR_READ_FOLLOWERS, AuthScope.CHANNEL_READ_ADS]

with open('json_schemas/message_input.json', 'r') as file:
    msg_schema = json.loads(file.read())

input_queue_name = 'text_input'

class TwitchWorker:
    guild_id: str
    twitch: Twitch
    helper: UserAuthenticationStorageHelper
    user: TwitchUser | None
    eventsub: EventSubWebsocket | None = None
    connected: bool
    redemption_list: list[str]
    subscriptions: bool
    cheers: bool
    follows: bool
    ads: bool
    ads_immediate: bool

    def __init__(self, guild_id: str, twitch_id: str, twitch_secret:str):
        self.guild_id = guild_id
        self.twitch = Twitch(twitch_id or '', twitch_secret)
        self.helper = UserAuthenticationStorageHelper(self.twitch, TARGET_SCOPES)
        self.redemption_list = []
        self.subscriptions = False
        self.cheers = False
        self.ads = False
        self.ads_immediate = False
        self.follows = False
        try:
            asyncio.run(self.helper.bind())
            self.connected = True
        except:
            print(f"Something went wrong with your Twitch credentials!")

    async def channel_redemptions_cb(self, data: ChannelPointsCustomRewardRedemptionAddEvent):
        for redemption in self.redemption_list:
            if data.event.reward.title == redemption:
                message = f'{data.event.user_name} has redeemed {data.event.reward.title}'
                if not data.event.user_input == '':
                    message += ' and says: ' + data.event.user_input
                self.send_message(message)

    def sub_tier(self, value: str):
        if value == '1000':
            return 1
        elif value == '2000':
            return 2
        elif value == '3000':
            return 3
        return 0
    
    async def subscriptions_cb(self, data: ChannelSubscribeEvent):
        if self.subscriptions and not data.event.is_gift:
            message = f'{data.event.user_name} has subscribed to {data.event.broadcaster_user_name} at tier {self.sub_tier(data.event.tier)}'
            self.send_message(message)

    async def gift_subscriptions_cb(self, data: ChannelSubscriptionGiftEvent):
        if self.subscriptions:
            user = data.event.user_name
            if data.event.is_anonymous:
                user = "An anonymous gifter"
            message = f'{user} has gifted {data.event.total} tier {self.sub_tier(data.event.tier)} subscriptions to {data.event.broadcaster_user_name}'
            self.send_message(message)

    async def resubscriptions_cb(self, data: ChannelSubscriptionMessageEvent):
        if self.subscriptions:
            message = f'{data.event.user_name} has subscribed to {data.event.broadcaster_user_name} for {data.event.duration_months} at tier {self.sub_tier(data.event.tier)} and says: {data.event.message.text}.'
            self.send_message(message)

    async def cheers_cb(self, data: ChannelCheerEvent):
        if self.cheers:
            user = data.event.user_name
            if data.event.is_anonymous:
                user = "An anonymous cheerer"
            message = f'{user} has cheered with {data.event.bits} and says: {data.event.message}'
            self.send_message(message)

    async def follows_cb(self, data: ChannelFollowEvent):
        if self.follows:
            message = f'{data.event.user_name} has followed the channel'
            self.send_message(message)

    async def ads_cb(self, data: ChannelAdBreakBeginEvent):
        if self.ads:
            self.send_message('An ad has begun on the channel', self.ads_immediate)
            timer = threading.Timer(data.event.duration_seconds, self.send_message, args=('Ads have ended', self.ads_immediate))
            timer.start()
            
    def send_message(self, message: str, immediate: bool = False):
        rabbit_connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost'))
        rabbit_channel = rabbit_connection.channel()
        input_queue_declare = rabbit_channel.queue_declare(input_queue_name, durable=True)
        priority = "normal"
        if immediate:
            priority = "immediate"
        msg = {
            "guildId": self.guild_id,
            "text": message,
            "priority": priority,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
        }
        validate(msg, msg_schema)
        rabbit_channel.basic_publish(
            exchange='',
            routing_key=input_queue_name,
            body=json.dumps(msg),
            properties=pika.BasicProperties(
                delivery_mode=pika.DeliveryMode.Persistent
            )
        )
        rabbit_connection.close()
    
    async def start(self):
        self.user = await first(self.twitch.get_users())
        if self.eventsub is None and self.user is not None:
            self.eventsub = EventSubWebsocket(self.twitch)
            self.eventsub.start()
            await self.eventsub.listen_channel_points_custom_reward_redemption_add(self.user.id, self.channel_redemptions_cb)
            await self.eventsub.listen_channel_subscribe(self.user.id, self.subscriptions_cb)
            await self.eventsub.listen_channel_subscription_gift(self.user.id, self.gift_subscriptions_cb)
            await self.eventsub.listen_channel_subscription_message(self.user.id, self.resubscriptions_cb)
            await self.eventsub.listen_channel_cheer(self.user.id, self.cheers_cb)
            await self.eventsub.listen_channel_follow_v2(self.user.id, self.user.id, self.follows_cb)
            await self.eventsub.listen_channel_ad_break_begin(self.user.id, self.ads_cb)
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                print('asdf')
                await self.eventsub.stop()
                quit()

if (__name__ == "__main__"):
    owo = TwitchWorker("324917760578682880", os.getenv("TWITCH_ID") or '', os.getenv("TWITCH_SECRET") or '')
    owo.ads = True
    owo.ads_immediate = True
    owo.subscriptions = True
    owo.redemption_list = ['Talk to Sapphire']
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
    # TODO: Make it so that when a familiar with twitch enabled joins, spin up a new thread. going to have to verify on twitch to get uesr_token.json through discord.js slash commands
