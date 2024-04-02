/** @type {import('./index.js').Command} */
import { Buffer } from "node:buffer";
import process from "node:process";
import pkg from "@discordjs/opus"
const { OpusEncoder } = pkg;
import { joinVoiceChannel, createAudioPlayer, createAudioResource, StreamType, AudioPlayerStatus, VoiceConnectionStatus, getVoiceConnection } from "@discordjs/voice";
import * as amqp from "amqplib";
import { ChannelType, VoiceChannel } from "discord.js";
import { Readable } from "stream"

const encoder = new OpusEncoder(48_000, 2);
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

export default {
	data: {
		name: 'awaken',
		description: 'Direct your familiar to join the voice chat you are in.',
	},
	async execute(interaction) {
		const prelim_connection = getVoiceConnection(interaction.guild.id)
		if (prelim_connection) {
			prelim_connection.disconnect()
		}

		const client = interaction.client;
		let user_channel = '';
		interaction.client.channels.cache.forEach((channel) => {
			if (channel.type == ChannelType.GuildVoice && channel.joinable) {	
				if (channel.members.has(interaction.user.id)) {
					user_channel = channel.id
				}
			}
		});
		if (user_channel == '') {
			await interaction.reply({content: 'Join a voice channel your familiar can join first!', ephemeral: true})
			return
		}
		
		const channel = await client.channels.fetch(user_channel);
		await interaction.reply({content: 'Joined ' + channel.name, ephemeral: true})
		const audioBuffers = {};
		const voiceSubs = {};
		if (channel instanceof VoiceChannel) {
			const connection = joinVoiceChannel({
				channelId: channel.id,
				guildId: channel.guild.id,
				adapterCreator: channel.guild.voiceAdapterCreator,
				selfDeaf: false,
			});
			const player = createAudioPlayer()
			const playerSub = connection.subscribe(player)
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

			const voiceOutQueue = await rabbitChannel.assertQueue('', { exclusive:true})
			rabbitChannel.bindQueue(voiceOutQueue.queue, voiceOutExchange, channel.guild.id)
			// Make it so that the consumer is only defined and started in one place.
			const consumer = await rabbitChannel.consume(voiceOutQueue.queue, function(msg){
				console.log("Voice line received for " + channel.guild.name)
				
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

			connection.on(VoiceConnectionStatus.Disconnected, () => {
				console.log('Disconnected from ' + channel.guild.name);
				rabbitChannel.unbindQueue(voiceOutQueue.queue, voiceOutExchange, channel.guild.id)
				rabbitChannel.cancel(consumer.consumerTag)
				connection.receiver.speaking.removeAllListeners('start')
				connection.receiver.speaking.removeAllListeners('end')
				playerSub.unsubscribe()
				player.removeAllListeners(AudioPlayerStatus.Idle);
				player.stop();
				connection.destroy()
			})
			process.on('SIGINT', function(){
				console.log('Disconnected from RabbitMQ and closing...');
				rabbitConnection.close();
				connection.disconnect();
				setTimeout(() => { process.exit(0) }, 1000);
			})
		}
	},
};
