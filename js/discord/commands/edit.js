import { ModalBuilder, TextInputStyle, ActionRowBuilder, TextInputBuilder, PermissionFlagsBits, SlashCommandBuilder, StringSelectMenuBuilder, ButtonStyle, StringSelectMenuOptionBuilder, ButtonBuilder } from 'discord.js';
import { createClient } from '@libsql/client'

const db = new createClient({
	url: "file:../local.db",
})

/** @type {import('./index.js').Command} */
export default {
	data: new SlashCommandBuilder()
		.setName('setup')
		.setDescription('Edit your familiar\'s details.')
		.setDefaultMemberPermissions(PermissionFlagsBits.BanMembers)
		.addStringOption(option =>
			option.setName('option')
				.setDescription('Option to edit')
				.addChoices(
					{ name: 'Basic', value: 'basic' },
					{ name: 'API Providers', value: 'providers'},
					{ name: 'API Keys', value: 'apikeys'}
				)
				.setRequired(true)
			)
	,
	async execute(interaction) {
		let row = await db.execute({
			sql: "SELECT * FROM familiars WHERE guildId = ?",
			args: [interaction.guildId],
		})
		if (row.rows.length == 0) {
			console.log('Cannot find guild ID')
			await db.execute({
				sql: "INSERT INTO familiars (guildId, name) VALUES (?,?)",
				args: [interaction.guildId, interaction.client.user.displayName]
			})
			row = await db.execute({
				sql: "SELECT * FROM familiars WHERE guildId = ?",
				args: [interaction.guildId]
			})
		}
		const familiar = row.rows[0]	
		if (interaction.options.getString('option') == 'basic') {
			const modal = new ModalBuilder()
				.setCustomId('basic-info')
				.setTitle('Edit ' + familiar.name)
			const nameInput = new TextInputBuilder()
				.setCustomId('name')
				.setLabel('Name')
				.setValue(familiar.name)
				.setStyle(TextInputStyle.Short);
			const chattiness = new TextInputBuilder()
				.setCustomId('chattiness')
				.setLabel('Chattiness')
				.setValue(familiar.chattiness.toString())
				.setStyle(TextInputStyle.Short);
			const promptInput = new TextInputBuilder()
				.setCustomId('prompt')
				.setLabel('Enter the prompt for your familiar')
				.setValue(familiar.prompt)
				.setStyle(TextInputStyle.Paragraph);
			const keywords = new TextInputBuilder()
				.setCustomId('keywords')
				.setLabel('Keywords for your familiar to respond to')
				.setPlaceholder('Seperate by comma')
				.setValue(familiar.keywords)
				.setStyle(TextInputStyle.Paragraph);
			
			const row1 = new ActionRowBuilder().addComponents(nameInput)
			const row2 = new ActionRowBuilder().addComponents(chattiness)
			const row3 = new ActionRowBuilder().addComponents(promptInput)
			const row4 = new ActionRowBuilder().addComponents(keywords)
			modal.addComponents(row1, row2, row3, row4)
			await interaction.showModal(modal)
		} else if (interaction.options.getString('option') == 'providers') {
			const transcriberSelect = new StringSelectMenuBuilder()
				.setCustomId('transcriber-menu')
				.setPlaceholder('Select provider for transcription')
				.addOptions(
					new StringSelectMenuOptionBuilder()
						.setDefault(familiar.transcriber == 'deepgram')
						.setLabel('Speech Transcriber: Deepgram')
						.setValue('deepgram'),
					new StringSelectMenuOptionBuilder()
						.setDefault(familiar.transcriber == 'openaiWhisper')
						.setLabel('Speech Transcriber: OpenAI Whisper')
						.setValue('openai'),
				);
			const llmSelect = new StringSelectMenuBuilder()
				.setCustomId('llm-menu')
				.setPlaceholder('Select provider for generating responses')
				.addOptions(
					new StringSelectMenuOptionBuilder()
						.setDefault(familiar.llm == 'anthropic')
						.setLabel('Text Generator: Anthropic')
						.setValue('anthropic'),
					new StringSelectMenuOptionBuilder()
						.setDefault(familiar.llm == 'openai')
						.setLabel('Text Generator: OpenAI')
						.setValue('openai'),
				)
			const ttsSelect = new StringSelectMenuBuilder()
				.setCustomId('tts-menu')
				.setPlaceholder('Select provider for your familiar\'s voice')
				.addOptions(
					new StringSelectMenuOptionBuilder()
						.setDefault(familiar.tts == 'azure')
						.setLabel('Text-To-Speech: Azure')
						.setValue('azure'),
				)
			const confirmButton = new ButtonBuilder()
				.setCustomId('confirm-providers')
				.setLabel('Next')
				.setStyle(ButtonStyle.Success)
			const row = new ActionRowBuilder().addComponents(transcriberSelect)
			const row2 = new ActionRowBuilder().addComponents(llmSelect)
			const row3 = new ActionRowBuilder().addComponents(ttsSelect)
			const row4 = new ActionRowBuilder().addComponents(confirmButton)
			await interaction.reply({
				content: "Select your choice of API",
				components: [row, row2, row3, row4],
				ephemeral: true
			})
		}
	}
};
