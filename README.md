This folder allows the reader to reproduce the results presented in the paper regarding the prompt engineering experiment. 

It highlights the following components:

1. Static public server that hosts the folder "prompt_engineering_experiment/experiment_photos/1_with_gaze"  gaze estimate dataset. It uses an fastapi uvicorn app and it makes it public using ngrok. To run this server:
uvicorn dataset_server:app --host 0.0.0.0 --port 8060 --reload
To make it public using ngrok:
ngrok http 8080
When running the docker container from the backend in vscode with its devcontainer.json, the ngrok authentication is set automatically with this command in the devcontainer.json:
	"postStartCommand": "ngrok config add-authtoken $(grep NGROK_AUTH_TOKEN /app/.env | cut -d '=' -f2)"

