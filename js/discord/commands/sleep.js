/** @type {import('./index.js').Command} */
import { getVoiceConnection } from '@discordjs/voice'
import { PermissionFlagsBits, SlashCommandBuilder } from 'discord.js';

export default {
	data: new SlashCommandBuilder()
		.setName('sleep')
		.setDescription('Let your famimliar rest and leave the voice channel.')
		.setDefaultMemberPermissions(PermissionFlagsBits.BanMembers)
	,
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
