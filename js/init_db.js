import { createClient } from '@libsql/client'

const db = new createClient({
	url: "file:../local.db",
})

db.execute('CREATE TABLE familiars (guildId TEXT PRIMARY KEY, name TEXT DEFAULT "Familiar", prompt TEXT DEFAULT "", chattiness TEXT DEFAULT "75", keywords TEXT DEFAULT "", transcriber TEXT DEFAULT "", llm TEXT DEFAULT "", tts TEXT DEFAULT "")')
db.execute('CREATE TABLE deepgram (guildId TEXT PRIMARY KEY, apikey TEXT DEFAULT "")')
db.execute('CREATE TABLE anthropic (guildId TEXT PRIMARY KEY, apikey TEXT DEFAULT "", model TEXT DEFAULT "", tempurature TEXT DEFAULT "")')
db.execute('CREATE TABLE openai (guildId TEXT PRIMARY KEY, apikey TEXT DEFAULT "", model TEXT DEFAULT "", tempurature TEXT DEFAULT "")')
db.execute('CREATE TABLE azure (guildId TEXT PRIMARY KEY, apikey TEXT DEFAULT "", region TEXT DEFAULT "", voice TEXT DEFAULT "")')
db.execute('CREATE TABLE history (guildId TEXT, msg TEXT DEFAULT "", image TEXT DEFAULT "", role TEXT DEFAULT "user", timestamp TEXT DEFAULT "")')
db.execute('CREATE TABLE message_queue (guildId TEXT, msg TEXT DEFAULT "", image TEXT DEFAULT "", role TEXT DEFAULT "user", timestamp TEXT DEFAULT "")')
db.execute('CREATE TABLE users_speaking (guildId TEXT, user TEXT)')
