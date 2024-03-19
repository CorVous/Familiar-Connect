import type { Command } from './index';

export default {
	data: {
		name: 'ping',
		description: 'Ping!',
	},
	async execute(interaction) {
		await interaction.reply('Pong!');
	},
} satisfies Command;
