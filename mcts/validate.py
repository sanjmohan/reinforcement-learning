from kaggle_environments import evaluate, make, utils
import sys


out = sys.stdout
submission = utils.read_file("submission.py")
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make("connectx", configuration={"actTimeout": 5})
print("Validatiing - this may take a minute...")
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")

