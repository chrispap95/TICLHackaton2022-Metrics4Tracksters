# TICLHackaton2022-Metrics4Tracksters

## Initial instructions

To connect use the following command:
```bash
ssh -L localhost:8XXX:localhost:8XXX <user>@lxplus.cern.ch
```
where `8XXX` should be a free port (pick any number you like) and `<user>` is your username.

In lxplus, move to the directory you want to use and execute the following as the initial setup:
```bash
git clone https://github.com/chrispap95/TICLHackaton2022-Metrics4Tracksters.git
cd TICLHackaton2022-Metrics4Tracksters
source setup.sh
python3 -m venv --copies ticlenv
source ticlenv/bin/activate
python -m pip install --no-cache-dir awkward
python -m pip install --no-cache-dir uproot
ipython kernel install --user --name=ticlenv
```

## Connecting instructions
To reconnect, use the same ssh command and from the same directory execute:
```bash
source setup.sh
source ticlenv/bin/activate
```

To initialize jupyter:
```bash
source jupy.sh 8XXX
```
To connect, open your browser and connect to the address given by the script.
You should create a new notebook with kernel `ticlenv`.
