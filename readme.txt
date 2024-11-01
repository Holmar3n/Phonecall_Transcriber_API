Systemet använder sig utav 2 modeller, en för transkriberingen och en för röstanalys.
API:et tar emot en filväg och ljudfilerna förbehandlas för att fungera bättre i modellerna.
Modellerna skapar segment som sedan matchas ihop med hjälp av match_segments() funktionen.
Med hjälp av identify_roles_by_keywords() funktionen så bestäms vilka roller Talare 1 och Talare 2 får (agent eller customer).
Sen görs den slutgiltiga listan om till en json fil som sedan skickas tillbaka.
Finns även ett log system som skapar en logs mapp om den inte redan finns och för varje dag skapas en ny fil där felmeddelanden skrivs in.

Installationer/modeller:
Torch
pyannote.audio
rapidfuzz
numpy
pydub
whisper  https://github.com/openai/whisper.git
flask
noisereduce

Behörighet:
Pyannote kräver att man skapar en token på huggingface och accepterar deras villkor.

övrigt:
Skickade med några ljudfiler att testa på. Några är bra medans andra är mindre bra.
I funktionen identify_roles_by_keywords() behövs sökvägen till keywords_data.json


PS Har bara kunnat köra koden på google colabs betalversion då modellerna kräver för mycket prestanda