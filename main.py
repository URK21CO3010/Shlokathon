phone_number = "+916383006581"              # REPLACE WITH YOUR PHONE NUMBER
server_channel_id = 1217350511553155113     # REPLACE WITH YOUR SERVER CHANNEL

from openai import OpenAI
import pandas as pd
import discord
from threading import Thread
from time import sleep
from twilio.rest import Client
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def get_response(client, training_data):
    response = client.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages = training_data,
    temperature=1,
    top_p=1,
    frequency_penalty=0.6,
    presence_penalty=0
    )

    return response

def get_training_data(training_file):
    
    data = pd.read_excel(training_file)

    dict_list = data.to_dict(orient='records')

    return dict_list

def update_training_data(old_data, new_data):

    old_data.extend(new_data)

    updated_data = pd.DataFrame(old_data)

    updated_data.to_excel("Speech Training.xlsx", index=False)



def initialize_training_file(initial_data):

    data = pd.DataFrame(initial_data)
    data.to_excel("Speech Training.xlsx", index = False)


def start_maintenance_assistant():
    
    ai_client = OpenAI(api_key = "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")

    data = {'role': ['system'], 'content': ['You are an assistant who helps with malfunctioning equipment.']}

    initialize_training_file(initial_data = data)



    maintenance_assistant_client = discord.Client(intents = discord.Intents.all())

    @maintenance_assistant_client.event
    async def on_ready():
        
        channel = maintenance_assistant_client.get_channel(server_channel_id)


    @maintenance_assistant_client.event
    async def on_message(message):
            
        if not message.author.bot:
        
            training_data = get_training_data(training_file = "Speech Training.xlsx")

            chat_prompt = message.content

            new_data = [{'role': 'user', 'content': chat_prompt}]

            update_training_data(old_data = training_data, new_data = new_data)

            training_data = get_training_data(training_file = "Speech Training.xlsx")

            response = get_response(client = ai_client, training_data = training_data).choices[0].message.content

            new_data = [{'role': 'assistant', 'content': response}]

            update_training_data(old_data = training_data, new_data = new_data)

            await message.channel.send(response)

    maintenance_assistant_client.run("MTIxOTIwNzIzMjk1NzMxNzE0MA.GzZ4XK.bLuqgsrIZkuEHK9hgPIu5pK88hJZGkUND35mvw")




def start_timer(time):
    global responded
    while time > 0:
        time -= 1
        sleep(1)
    if not responded:
        alert_via_call(phone_number = phone_number)


def alert_via_server(channel_id, alert_message):

    discord_client = discord.Client(intents = discord.Intents.all())


    @discord_client.event
    async def on_ready():
        await discord_client.get_channel(channel_id).purge()
        global alert_message, responded
        alert_message = await discord_client.get_channel(channel_id).send(alert_message + "React with a 👍")
        reply_time = 10
        responded = False
        timer_thread = Thread(target = start_timer, args = [reply_time])
        timer_thread.start()
        

    @discord_client.event
    async def on_raw_reaction_add(payload):
        
        global alert_message, responded

        if payload.user_id != discord_client.user.id:
            if payload.message_id == alert_message.id:
                if payload.emoji.name == '👍':
                    responded = True
                    channel_id = 1217350511553155113
                    await discord_client.get_channel(channel_id).send("Maintenance Initiated")


    discord_client.run("MTIxNzM1MDA3MzQwNzYzNTQ2Ng.GaFc8O.DWoUfId5oo6V1pU1AHGJe_1AqdqrxdXapbPd2g")

def alert_via_call(phone_number = "+916383006581"):
    account_sid = "ACd2ae0efc35de4366e24c5c6e84245faf"
    auth_token = "6aa3f162ceb8e183d94a27162ef1b2a9"
    client = Client(account_sid, auth_token)

    call = client.calls.create(
    url="http://demo.twilio.com/docs/voice.xml",
    to=str(phone_number),
    from_="+12097574309"
    )


def get_anomalous_sensors(file_path):

    data = pd.read_csv(file_path)
    
    data_drop = data.drop(columns=['Unnamed: 0', 'timestamp','sensor_15'])

    data_interpolated = data_drop.interpolate(method='linear', limit_direction='forward', axis=0)

    X = data_interpolated.drop(columns=['machine_status'])
    y = data_interpolated['machine_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    isolation_forest.fit(X_train_scaled)

    anomaly_scores = isolation_forest.decision_function(X_test_scaled)

    threshold = np.percentile(anomaly_scores, 100 * 0.1)

    anomalies = X_test[anomaly_scores < threshold]

    anomalous_sensors = anomalies.mean().idxmax()

    anomaly_indices = anomaly_scores < threshold
    anomalies = X_test[anomaly_indices]

    num_sensors = 5
    anomalous_sensors = anomalies.mean().nlargest(num_sensors).index
    anomalous_sensors = anomalous_sensors.tolist()
    anomalous_sensors.sort()

    return anomalous_sensors



maintenance_assistant_thread = Thread(target = start_maintenance_assistant)
maintenance_assistant_thread.start()



anomalous_sensors = get_anomalous_sensors('sampled_sensor.csv')

alert_message = "Anomalous Sensors : \n"

for anomalous_sensor in anomalous_sensors:
    alert_message += anomalous_sensor + "\n"


if len(anomalous_sensors) > 0:
    alert_via_server(channel_id = server_channel_id, alert_message = alert_message)


