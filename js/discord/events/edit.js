import { ActionRowBuilder, ButtonBuilder, ButtonStyle, Events, ModalBuilder, StringSelectMenuBuilder, StringSelectMenuOptionBuilder, TextInputBuilder, TextInputStyle } from 'discord.js';
import { createClient } from '@libsql/client'

const db = new createClient({
	url: "file:../local.db",
})

const AZURE_VOICES = [
    "en-US-DavisNeural",
    "en-US-TonyNeural",
    "en-US-JasonNeural",
    "en-US-GuyNeural",
    "en-US-JaneNeural",
    "en-US-NancyNeural",
    "en-US-JennyNeural",
    "en-US-AriaNeural",
    "en-US-AmberNeural",
]
const API_PLACEHOLDER = "*****"

/** @type {import('./index.js').Event<Events.InteractionCreate>} */
export default {
	name: Events.InteractionCreate,
	once: false,
	async execute(interaction) {
        if (interaction.isModalSubmit()) {
            if (interaction.customId === 'basic-info') {
                const name = interaction.fields.getTextInputValue('name');
                const chattinessValue = parseInt(interaction.fields.getTextInputValue('chattiness'));
                let chattiness = "75"
                if (chattinessValue <= 100 && chattinessValue >= 0) {
                    chattiness = chattinessValue.toString()
                }
                const prompt = interaction.fields.getTextInputValue('prompt');
                const keywords = interaction.fields.getTextInputValue('keywords');
                await db.execute({ 
                    sql: "UPDATE familiars SET name = ?, chattiness = ?, prompt = ?, keywords = ? WHERE guildId = ?",
                    args: [name, chattiness, prompt, keywords, interaction.guildId]
                })
                interaction.reply({content: 'Your familiar has been updated!', ephemeral: true});
            } else if (interaction.customId === 'apikey-modal') {
                const row = await db.execute({ sql: "SELECT transcriber, llm, tts FROM familiars WHERE guildId = ?", args:[interaction.guildId]})
                const familiar = row.rows[0]
                
                const transcribeKey = interaction.fields.getTextInputValue('transcribe-key').trim()
                const llmKey = interaction.fields.getTextInputValue('llm-key').trim()
                const ttsKey = interaction.fields.getTextInputValue('tts-key').trim()
                const azureRegion = interaction.fields.getTextInputValue('azure-region').trim()
                
                if (!transcribeKey.includes(API_PLACEHOLDER)) {
                    await db.execute({ sql: "UPDATE "+familiar.transcriber+" SET apiKey=? WHERE guildId=?", args: [transcribeKey, interaction.guildId] })
                }
                if (!llmKey.includes(API_PLACEHOLDER)) {
                    await db.execute({ sql: "UPDATE "+familiar.llm+" SET apiKey=? WHERE guildId=?", args: [llmKey, interaction.guildId] })
                }
                if (!ttsKey.includes(API_PLACEHOLDER)) {
                    await db.execute({ sql: "UPDATE "+familiar.tts+" SET apiKey=? WHERE guildId=?", args: [ttsKey, interaction.guildId] })
                }
                if (azureRegion) {
                    await db.execute({ sql: "UPDATE azure SET region=? WHERE guildId=?", args: [azureRegion, interaction.guildId] })
                }
                // TODO: Verify all api keys that are submitted
                
                interaction.update({content: 'You are done with setting up your API providers!', components: []})
            }
        } else if (interaction.isStringSelectMenu()) {
            if (interaction.customId == 'transcriber-menu') {
                await db.execute({ sql: "UPDATE familiars SET transcriber = ? WHERE guildId = ?", args: [interaction.values[0], interaction.guildId] })
            } else if (interaction.customId == 'llm-menu') {
                await db.execute({ sql: "UPDATE familiars SET llm = ? WHERE guildId = ?", args: [interaction.values[0], interaction.guildId] })
            } else if (interaction.customId == 'tts-menu') {
                await db.execute({ sql: "UPDATE familiars SET tts = ? WHERE guildId = ?", args: [interaction.values[0], interaction.guildId] })
            } else if (interaction.customId == 'anthropic-model') {
                await db.execute({ sql: "UPDATE anthropic SET model = ? WHERE guildId = ?", args: [interaction.values[0], interaction.guildId]})
            } else if (interaction.customId == 'openai-model') {
                await db.execute({ sql: "UPDATE openai SET model = ? WHERE guildId = ?", args: [interaction.values[0], interaction.guildId]})
            } else if (interaction.customId == 'anthropic-tempurature') {
                await db.execute({ sql: "UPDATE anthropic SET tempurature = ? WHERE guildId = ?", args: [interaction.values[0], interaction.guildId]})
            } else if (interaction.customId == 'openai-tempurature') {
                await db.execute({ sql: "UPDATE anthropic SET tempurature = ? WHERE guildId = ?", args: [interaction.values[0], interaction.guildId]})
            } else if (interaction.customId == 'azure-voice') {
                await db.execute({ sql: "UPDATE azure SET voice = ? WHERE guildId = ?", args: [interaction.values[0], interaction.guildId]})
            }
            interaction.deferUpdate()
        } else if (interaction.isButton()) {
            if (interaction.customId == 'confirm-providers') {
                const row = await db.execute({ sql: "SELECT transcriber, llm, tts FROM familiars WHERE guildId = ?", args:[interaction.guildId]})
                const familiar = row.rows[0]
                if (familiar.transcribe === '' || familiar.llm === '' || familiar.tts === '') {
                    await interaction.update({content: 'Select your choice of API\n***You must fill each dropdown!***'})
                    return;
                }
                let transcribeRow = await db.execute({sql: "SELECT * FROM '"+familiar.transcriber+"' WHERE guildId = ?", args: [interaction.guildId]})
                if (transcribeRow.rows.length == 0) {
                    await db.execute({sql: "INSERT INTO '"+familiar.transcriber+"' (guildId) VALUES (?)", args: [interaction.guildId]})
                    transcribeRow = await db.execute({sql: "SELECT * FROM '"+familiar.transcriber+"' WHERE guildId = ?", args: [interaction.guildId]})
                }
                const transcribeData = transcribeRow.rows[0]
                let llmRow = await db.execute({sql: "SELECT * FROM '"+familiar.llm+"' WHERE guildId = ?", args: [interaction.guildId]})
                if (llmRow.rows.length == 0) {
                    await db.execute({sql: "INSERT INTO '"+familiar.llm+"' (guildId) VALUES (?)", args: [interaction.guildId]})
                    llmRow = await db.execute({sql: "SELECT * FROM '"+familiar.llm+"' WHERE guildId = ?", args: [interaction.guildId]})
                }
                const llmData = llmRow.rows[0]
                let ttsRow = await db.execute({sql: "SELECT * FROM '"+familiar.tts+"' WHERE guildId = ?", args: [interaction.guildId]})
                if (ttsRow.rows.length == 0) {
                    await db.execute({sql: "INSERT INTO '"+familiar.tts+"' (guildId) VALUES (?)", args: [interaction.guildId]})
                    ttsRow = await db.execute({sql: "SELECT * FROM '"+familiar.tts+"' WHERE guildId = ?", args: [interaction.guildId]})
                }
                const ttsData = ttsRow.rows[0]
                
                const anthropicModels = new StringSelectMenuBuilder()
                    .setCustomId('anthropic-model')
                    .setPlaceholder('Select Anthropic LLM Model to use')
                    .addOptions(
                        new StringSelectMenuOptionBuilder()
                            .setLabel('Model: Claude 3 Opus')
                            .setDescription('Anthropics most powerful model')
                            .setDefault(llmData.model == 'claude-3-opus-20240229')
                            .setValue('claude-3-opus-20240229'),
                        new StringSelectMenuOptionBuilder()
                            .setLabel('Model: Claude 3 Sonnet')
                            .setDescription('Anthropic\'s medium sized model')
                            .setDefault(llmData.model == 'claude-3-sonnet-20240229')
                            .setValue('claude-3-sonnet-20240229'),
                        new StringSelectMenuOptionBuilder()
                            .setLabel('Model: Claude 3 Haiku')
                            .setDescription('Anthropics fastest model')
                            .setDefault(llmData.model == 'claude-3-haiku-20240307')
                            .setValue('claude-3-haiku-20240307')
                    );
                const anthropicModelRow = new ActionRowBuilder().addComponents(anthropicModels)
                
                const tempurature = new StringSelectMenuBuilder()
                    .setCustomId(familiar.llm + '-tempurature')
                    .setPlaceholder('Select Tempurature, higher is more creative')
                    .addOptions(
                        new StringSelectMenuOptionBuilder()
                            .setLabel('Tempurature: Lowest')
                            .setDefault(llmData.tempurature == '0')
                            .setValue('0'),
                        new StringSelectMenuOptionBuilder()
                            .setLabel('Tempurature: Low')
                            .setDefault(llmData.tempurature == '0.25')
                            .setValue('0.25'),
                        new StringSelectMenuOptionBuilder()
                            .setLabel('Tempurature: Medium')
                            .setDefault(llmData.tempurature == '0.5')
                            .setValue('0.5'),
                        new StringSelectMenuOptionBuilder()
                            .setLabel('Tempurature: High')
                            .setDefault(llmData.tempurature == '0.75')
                            .setValue('0.75'),
                        new StringSelectMenuOptionBuilder()
                            .setLabel('Tempurature: Highest')
                            .setDefault(llmData.tempurature == '1.0')
                            .setValue('1.0')
                    )
                const tempuratureRow = new ActionRowBuilder().addComponents(tempurature)
                
                let azureVoiceOptions = []
                AZURE_VOICES.forEach((voice) => {
                    const voiceOption = new StringSelectMenuOptionBuilder()
                        .setLabel('Azure Voice: ' + voice)
                        .setDefault(ttsData.voice == voice)
                        .setValue(voice)
                    azureVoiceOptions.push(voiceOption)
                })
                const azureVoice = new StringSelectMenuBuilder()
                    .setCustomId('azure-voice')
                    .setPlaceholder('Select Azure voice')
                    .addOptions(azureVoiceOptions)
                const azureVoiceRow = new ActionRowBuilder().addComponents(azureVoice)

                const confirmApiSettingsButton = new ButtonBuilder()
                    .setCustomId('confirm-api-settings')
                    .setLabel('Next')
                    .setStyle(ButtonStyle.Success)
                const confirmButtonRow = new ActionRowBuilder().addComponents(confirmApiSettingsButton)
                
                let nextSettings = []
                if (familiar.llm == 'anthropic') {
                    nextSettings.push(anthropicModelRow)
                }
                nextSettings.push(tempuratureRow)
                if (familiar.tts == 'azure') {
                    nextSettings.push(azureVoiceRow)
                }
                nextSettings.push(confirmButtonRow)
                await interaction.update({ content: 'Enter API settings', components: nextSettings })
            } else if (interaction.customId === 'confirm-api-settings') {
                const row = await db.execute({ sql: "SELECT transcriber, llm, tts FROM familiars WHERE guildId = ?", args:[interaction.guildId]})
                const familiar = row.rows[0]
                let transcriberRow = await db.execute({sql: "SELECT * FROM '"+familiar.transcriber+"' WHERE guildId = ?", args: [interaction.guildId]})
                if (transcriberRow.rows.length == 0) {
                    await db.execute({sql: "INSERT INTO '"+familiar.transcriber+"' (guildId) VALUES (?)", args: [interaction.guildId]})
                    transcriberRow = await db.execute({sql: "SELECT * FROM '"+familiar.transcriber+"' WHERE guildId = ?", args: [interaction.guildId]})
                }
                const transcriberData = transcriberRow.rows[0]
                let llmRow = await db.execute({sql: "SELECT * FROM '"+familiar.llm+"' WHERE guildId = ?", args: [interaction.guildId]})
                if (llmRow.rows.length == 0) {
                    await db.execute({sql: "INSERT INTO '"+familiar.llm+"' (guildId) VALUES (?)", args: [interaction.guildId]})
                    llmRow = await db.execute({sql: "SELECT * FROM '"+familiar.llm+"' WHERE guildId = ?", args: [interaction.guildId]})
                }
                const llmData = llmRow.rows[0]
                let ttsRow = await db.execute({sql: "SELECT * FROM '"+familiar.tts+"' WHERE guildId = ?", args: [interaction.guildId]})
                if (ttsRow.rows.length == 0) {
                    await db.execute({sql: "INSERT INTO '"+familiar.tts+"' (guildId) VALUES (?)", args: [interaction.guildId]})
                    ttsRow = await db.execute({sql: "SELECT * FROM '"+familiar.tts+"' WHERE guildId = ?", args: [interaction.guildId]})
                }
                const ttsData = ttsRow.rows[0]
                if (ttsData.voice == '' || llmData.model == '' || llmData.tempurature == '') {
                    await interaction.update({content: 'Enter API settings\n***You must fill each dropdown!***'})
                    return;
                }
                
                const modal = new ModalBuilder()
                    .setCustomId('apikey-modal')
                    .setTitle('API Key Entry')

                const transcribeInput = new TextInputBuilder()
                    .setCustomId('transcribe-key')
                    .setLabel('API Key for ' + familiar.transcriber)
                    .setValue(transcriberData.apikey ? API_PLACEHOLDER : '')
                    .setStyle(TextInputStyle.Short)
                const row1 = new ActionRowBuilder().addComponents(transcribeInput)
                modal.addComponents(row1)
                
                const llmInput = new TextInputBuilder()
                    .setCustomId('llm-key')
                    .setLabel('API Key for ' + familiar.llm)
                    .setValue(llmData.apikey ? API_PLACEHOLDER : '')
                    .setStyle(TextInputStyle.Short)
                const row2 = new ActionRowBuilder().addComponents(llmInput)
                modal.addComponents(row2)
                
                const ttsInput = new TextInputBuilder()
                    .setCustomId('tts-key')
                    .setLabel('API Key for ' + familiar.tts)
                    .setValue(ttsData.apikey ? API_PLACEHOLDER : '')
                    .setStyle(TextInputStyle.Short)
                const row3 = new ActionRowBuilder().addComponents(ttsInput)
                modal.addComponents(row3)
                
                if (familiar.transcriber == 'azure' || familiar.tts == 'azure') {
                    const regionRow = await db.execute({sql: "SELECT region FROM azure WHERE guildId = ?", args: [interaction.guildId]})
                    const region = regionRow.rows[0]
                    const azureRegion = new TextInputBuilder()
                        .setCustomId('azure-region')
                        .setLabel('Azure Region')
                        .setValue(region.region)
                        .setStyle(TextInputStyle.Short)
                    const rowAzure = new ActionRowBuilder().addComponents(azureRegion)
                    modal.addComponents(rowAzure)
                }
                
                await interaction.showModal(modal)
            }
        }
	},
};
