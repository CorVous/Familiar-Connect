/** @type {import('./index.js').Command} */
import { getVoiceConnection } from '@discordjs/voice'

export default {
	data: {
		name: 'sleep',
		description: 'Let your famimliar rest and leave the voice channel.',
	},
	async execute(interaction) {
		const connection = getVoiceConnection(interaction.guild.id)
		if (connection) {
			connection.disconnect()
			await interaction.reply({content: 'Your familiar is now asleep.', ephemeral: true})
		} else {
			await interaction.reply({content: 'Your familiar is already asleep!', ephemeral: true})
		}
	},
};
