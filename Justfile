ssh:
	ssh -t gaskin@login01.lisc.univie.ac.at

create-env:
	ssh -t gaskin@login01.lisc.univie.ac.at micromamba create -f train-clustered/train-clustered.yml -y

jupyter:
	ssh -t gaskin@login01.lisc.univie.ac.at tmux new-session -d -s jupyter01 /lisc/user/gaskin/.local/bin/micromamba run -n train-clustered jupyter lab --ip 0.0.0.0 --port 8888 --notebook-dir=~/train-clustered --no-browser --LabApp.token='' -y
