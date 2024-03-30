import { Buffer } from "node:buffer";
import process from "node:process";
import { URL } from "node:url";
import pkg from "@discordjs/opus"
const { OpusEncoder } = pkg;
import { joinVoiceChannel, createAudioPlayer, createAudioResource, StreamType, AudioPlayerStatus } from "@discordjs/voice";
import * as amqp from "amqplib";
import { Client, Events, GatewayIntentBits, VoiceChannel } from "discord.js";
import { loadCommands, loadEvents } from "./util/loaders.js";
import { registerEvents } from "./util/registerEvents.js";
import { Readable } from "stream"

const encoder = new OpusEncoder(48_000, 2);
const voiceEncoder = new OpusEncoder(16_000, 1)

// Connect to rabbitmq
const rabbitConnection = await amqp.connect("amqp://localhost");
const rabbitChannel = await rabbitConnection.createChannel();
const voiceInExchange = "voice_buffer";
const speakingExchange = "user_speaking";
const voiceOutExchange = "audio_output";
const CURRENT_TRANSCRIBER = 'deepgram';
rabbitChannel.assertExchange(speakingExchange, "topic", {
  durable: true,
});
rabbitChannel.assertExchange(voiceInExchange, "topic", {
  durable: true,
});
rabbitChannel.assertExchange(voiceOutExchange, "topic", {
  durable: false,
});
const voiceOutQueue = await rabbitChannel.assertQueue('', { exclusive:true})

// Initialize the client
const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildVoiceStates,
    GatewayIntentBits.GuildMessages,
  ],
});

// Load the events and commands
const events = await loadEvents(new URL("events/", import.meta.url));
const commands = await loadCommands(new URL("commands/", import.meta.url));

// Register the event handlers
registerEvents(commands, events, client);

// Login to the client
void client.login(process.env.DISCORD_TOKEN);

// listen for the client to be ready
client.once(Events.ClientReady, async (cli) => {
  console.log(`Ready! Logged in as ${cli.user.tag}`);
  const audioBuffers = {};
  const voiceSubs = {}
  const channel = await client.channels.fetch("931844044680749076");
  if (channel instanceof VoiceChannel) {
    const connection = joinVoiceChannel({
      channelId: channel.id,
      guildId: channel.guild.id,
      adapterCreator: channel.guild.voiceAdapterCreator,
      selfDeaf: false,
    });
    const player = createAudioPlayer()
    connection.subscribe(player)
    let speaking = new Set()
    let pausing
    let unpausing
    let lastPaused = 0
    console.log("Connected to channel " + channel.name);
    connection.receiver.speaking.on("start", (uid) => {
      if (speaking.size == 0) {
        if (unpausing) {
          clearTimeout(unpausing)
        }
        pausing = setTimeout(() => {
          lastPaused = new Date()
          player.pause()
        }, 3000)
      }
      speaking.add(uid)
      const speaking_key = channel.guild.id + ".discord-" + uid;
      rabbitChannel.publish(speakingExchange, speaking_key, Buffer.from([true]), { persistent: true })

      const user = client.users.cache.get(uid);
      voiceSubs[uid] = connection.receiver.subscribe(uid);
      voiceSubs[uid].on("data", (chunk) => {

        const audioChunk = encoder.decode(chunk);
        if (Array.isArray(audioBuffers[uid])) {
          audioBuffers[uid].push(audioChunk);
        } else {
          const newBuffer = [];
          newBuffer.push(audioChunk);
          audioBuffers[uid] = newBuffer;
        }
      });
    });
    connection.receiver.speaking.on("end", (uid) => {
      speaking.delete(uid)
      if (speaking.size == 0) {
        if (pausing) {
          clearTimeout(pausing)
        }
        unpausing = setTimeout(() => {
          if (player.state.status == AudioPlayerStatus.Paused) {
            const currentTime = new Date()
            if (lastPaused != 0) {
              const delta = (currentTime.getTime() - lastPaused.getTime()) / 1000
              lastPaused = 0
              if (delta > 15) {
                console.log('Talking for too long, discarding voicelines')
                voiceLines = []
                player.stop()
              }
            } else {
              player.unpause()
            }
          }
        }, 2000)
      }
      voiceSubs[uid].destroy()
      
      const speaking_key = channel.guild.id + ".discord-" + uid;
      rabbitChannel.publish(speakingExchange, speaking_key, Buffer.from([false]), { persistent: true })
      
      const user = client.users.cache.get(uid);
      const audioBuffer = Buffer.concat(audioBuffers[uid]);
      const key = CURRENT_TRANSCRIBER + "." + channel.guild.id + "." + user.username.split('.').join('');
      rabbitChannel.publish(voiceInExchange, key, audioBuffer, {persistent: true});
      audioBuffers[uid] = [];
      
      console.log('Sent audio from ' + user.username);
    });
    
    // Send back audio! 
    let voiceLines = []
    player.on(AudioPlayerStatus.Idle, () => {
      console.log('idle...')
      if (voiceLines.length > 0) {
        console.log('playing next voice line')
        player.play(voiceLines.shift())
      } else if (voiceLines.length == 0) { 
        const speaking_key = channel.guild.id + ".discord-familiar";
        rabbitChannel.publish(speakingExchange, speaking_key, Buffer.from([false]), { persistent: true })
      }
    })
    
    rabbitChannel.bindQueue(voiceOutQueue.queue, voiceOutExchange, '#')
    rabbitChannel.consume(voiceOutQueue.queue, function(msg){
      console.log("Voice line received for " + msg.fields.routingKey)
      
      const voiceLine = new Readable()
      voiceLine._read = () => {}
      voiceLine.push(msg.content)
      const audio_resource = createAudioResource(voiceLine, { inputType: StreamType.Raw })
      
      if (player.state.status == 'idle') {
        player.play(audio_resource)
        const speaking_key = channel.guild.id + ".discord-familiar";
        rabbitChannel.publish(speakingExchange, speaking_key, Buffer.from([true]), { persistent: true })
      } else {
        voiceLines.push(audio_resource)
      }
      if (speaking.size > 0) {
        player.pause()
      }
    }, { noAck: true })

    process.on('SIGINT', function(){
      connection.destroy();
      rabbitConnection.close();
      setTimeout(() => { process.exit(0) }, 1000)
    })
  }
});
