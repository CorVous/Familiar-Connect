import process from 'node:process';
import { URL } from 'node:url';
import { Client, GatewayIntentBits, VoiceChannel, Events } from 'discord.js';
import { joinVoiceChannel } from '@discordjs/voice'
import { loadCommands, loadEvents } from './util/loaders.js';
import { registerEvents } from './util/registerEvents.js';

// Initialize the client
const client = new Client({ intents: [GatewayIntentBits.Guilds] });

// Load the events and commands
const events = await loadEvents(new URL('events/', import.meta.url));
const commands = await loadCommands(new URL('commands/', import.meta.url));

// Register the event handlers
registerEvents(commands, events, client);

// Login to the client
void client.login(process.env.DISCORD_TOKEN);

// listen for the client to be ready
client.once(Events.ClientReady, async (c) => {
  console.log(`Ready! Logged in as ${c.user.tag}`);
  const channel = await client.channels.fetch('324917760578682881')
  if (channel instanceof VoiceChannel) {
    const connection = joinVoiceChannel({
      channelId: channel.id,
      guildId: channel.guild.id,
      adapterCreator: channel.guild.voiceAdapterCreator,
    });
    console.log('Connected to channel ' + channel.name)
    await setTimeout(()=>{
        connection.destroy()
        console.log('Disconnected from channel')
    }, 5000)
  }
});
