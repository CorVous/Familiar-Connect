import Buffer from "node:buffer";
import process from "node:process";
import { URL } from "node:url";
import pkg from "@discordjs/opus";
import { joinVoiceChannel } from "@discordjs/voice";
import * as amqp from "amqplib";
import { Client, Events, GatewayIntentBits, VoiceChannel } from "discord.js";
import { loadCommands, loadEvents } from "./util/loaders.js";
import { registerEvents } from "./util/registerEvents.js";

const { OpusEncoder } = pkg;
const encoder = new OpusEncoder(48_000, 2);

// Connect to rabbitmq
const rabbitConnection = await amqp.connect("amqp://localhost");
const rabbitChannel = await rabbitConnection.createChannel();
const rabbitExchange = "voice_buffer";

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
    console.log("Connected to channel " + channel.name);
    connection.receiver.speaking.on("start", (uid) => {
      const user = client.users.cache.get(uid);
      console.log(user.username + " is speaking");
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
      voiceSubs[uid].destroy()
      const user = client.users.cache.get(uid);
      const audioBuffer = Buffer.Buffer.concat(audioBuffers[uid]);
      const key = channel.guild.id + "." + user.username;
      rabbitChannel.assertExchange(rabbitExchange, "topic", {
        durable: false,
      });
      rabbitChannel.publish(rabbitExchange, key, audioBuffer);

      audioBuffers[uid] = [];
      console.log(user.username + " has stopped speaking");
    });
    setTimeout(() => {
      connection.destroy();
      rabbitConnection.close();
      console.log("Disconnected from channel");
    }, 600_000);
  }
});
